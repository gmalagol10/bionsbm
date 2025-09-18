"""
bionsbm

Copyright(C) 2021 fvalle1 & gmalagol10

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY
without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see < http: // www.gnu.org/licenses/>.
"""

import warnings
warnings.filterwarnings("ignore")
import functools
import os, sys
import logging

from graph_tool.all import load_graph, Graph, minimize_nested_blockmodel_dl

import numpy as np
import pandas as pd
import cloudpickle as pickle

from muon import MuData
from anndata import AnnData
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from scipy import sparse
from numba import njit
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:  # prevent adding multiple handlers
	ch = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)
######################################
import time
from muon import read_h5mu

class bionsbm():
	"""
	Class to run bionsbm
	"""
	def __init__(self, obj, label: Optional[str] = None, max_depth: int = 6, modality: str = "Mod1", saving_path: str = "results/myself", path_to_graph=None):
		"""
		Initialize a bionsbm self.

		This constructor sets up the graph representation of the input data
		(`AnnData` or `MuData`) and optionally assigns node types based on a label.

		Parameters
		----------
		obj : muon.MuData or anndata.AnnData
			Input data object. If `MuData`, all modalities are extracted; if `AnnData`,
			only the provided `modality` is used.
		label : str, optional
			Column in `.obs` used to assign document labels and node types.
			If provided, the graph is annotated accordingly.
		max_depth : int, default=6
			Maximum number of levels to save or annotate in the hierarchical self.
		modality : str, default="Mod1"
			Name of the modality to use when the input is `AnnData`.
		saving_path : str, default="results/myself"
			Base path for saving self outputs (graph, state, results).

		Notes
		-----
		- For `MuData`, multiple modalities are combined into a multi-branch graph.
		- If `label` is provided, a mapping is created to encode document/node types.
		- `self.g` (graph) and related attributes (`documents`, `words`, `keywords`)
		  are initialized by calling `self.make_graph(...)`.
		"""
		super().__init__()
		self.keywords: List = []
		self.nbranches: int = 1
		self.modalities: List[str] = []
		self.max_depth: int = max_depth
		self.obj: Any = obj
		self.saving_path: str = saving_path
		self.path_to_graph = path_to_graph

		if isinstance(obj, MuData):
			self.modalities=list(obj.mod.keys())   
			dfs=[obj[key].to_df().T for key in self.modalities]
			self.make_graph(dfs[0], dfs[1:])

		elif isinstance(obj, AnnData):
			self.modalities=[modality]
			self.make_graph(obj.to_df().T, [])

		if label:
			g_raw=self.g.copy()
			logger.info("Label found")
			metadata=obj[self.modalities[0]].obs
			mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(obj[self.modalities[0]].obs[label])))])
			inv_map = {v: k for k, v in mymap.items()}

			docs_type=[int(mymap[metadata.loc[doc][label]]) for doc in self.documents]
			types={}
			types["Docs"]=docs_type
			for i, key in enumerate(self.modalities):
				types[key]=[int(i+np.max(docs_type)+1) for a in range(0, obj[key].shape[0])]
			node_type = g_raw.new_vertex_property('int', functools.reduce(lambda a, b : a+b, list(types.values())))
			self.g = g_raw.copy()
		else:
			node_type=None
		self.node_type=node_type 

		
	def make_graph(self, df: pd.DataFrame, df_keyword_list: List[pd.DataFrame]) -> None:
		"""
		Build a heterogeneous graph from a main feature DataFrame and optional keyword/meta-feature DataFrames.

		This function constructs a bipartite (documents–words) or multi-branch
		graph (documents–words–keywords/meta-features) using the input matrices.
		If a cached graph file exists at ``self.saving_path``, it is loaded directly
		instead of rebuilding.

		Parameters
		----------
		df : pandas.DataFrame
			Main feature matrix with words/features as rows (index) and
			documents/samples as columns.
		df_keyword_list : list of pandas.DataFrame
			List of additional matrices (e.g., keywords, annotations, or meta-features).
			Each DataFrame must have the same columns as ``df`` (documents),
			and its rows will be treated as a separate feature branch.

		Notes
		-----
		- Each branch is assigned a unique ``kind`` index:
		  * 0 → documents
		  * 1 → main features (e.g., words/genes)
		  * 2, 3, ... → subsequent keyword/meta-feature branches
		- If a saved graph already exists at
		  ``{self.saving_path}_graph.xml.gz``, it will be loaded instead of recreated.
		- After graph construction, the graph is saved to disk in Graph-Tool format.

		Raises
		------
		ValueError
			If ``df`` and ``df_keyword_list`` cannot be aligned properly
			(e.g., inconsistent columns).
		"""
		if os.path.isfile(f"{self.saving_path}_graph.xml.gz") == True: 
			self.load_graph(filename=f"{self.saving_path}_graph.xml.gz")

		elif os.path.isfile(f"{self.path_to_graph}_graph.xml.gz") == True:
			self.load_graph(filename=f"{self.path_to_graph}_graph.xml.gz")

		else:  
			logger.info("Creating graph from multiple DataFrames")
			df_all = df.copy(deep =True)
			for ikey,df_keyword in enumerate(df_keyword_list):
				df_keyword = df_keyword.reindex(columns=df.columns)
				df_keyword.index = ["".join(["#" for _ in range(ikey+1)])+str(keyword) for keyword in df_keyword.index]
				df_keyword["kind"] = ikey+2
				df_all = pd.concat((df_all,df_keyword), axis=0)
   
			def get_kind(word):
				return 1 if word in df.index else df_all.at[word,"kind"]
   
			self.nbranches = len(df_keyword_list)
		   
			self.make_graph_single(df_all.drop("kind", axis=1, errors='ignore'), get_kind)

			folder = os.path.dirname(self.saving_path)
			Path(folder).mkdir(parents=True, exist_ok=True)
			self.save_graph(filename=f"{self.saving_path}_graph.xml.gz")


	def make_graph_single(self, df: pd.DataFrame, get_kind) -> None:

		"""
		Construct a graph-tool graph from a single feature matrix.

		This method builds a bipartite or multi-branch graph from the given
		DataFrame, where columns represent documents/samples and rows represent
		features (e.g., words, genes, or keywords). Vertices are created for
		both documents and features, and weighted edges connect documents to
		their features.

		Parameters
		----------
		df : pandas.DataFrame
			Feature matrix with rows as features (words, genes, or keywords)
			and columns as documents/samples. The values must be numeric and
			represent counts or weights of feature occurrences.
		get_kind : callable
			Function that takes a feature name (row index from ``df``) and
			returns an integer specifying the vertex kind:
			- 0 → document nodes
			- 1 → main feature nodes
			- 2, 3, ... → keyword/meta-feature branch nodes

		Notes
		-----
		- The constructed graph is undirected.
		- Vertices are annotated with two properties:
		  * ``name`` (string): document or feature name.
		  * ``kind`` (int): node type (document, word, or keyword branch).
		- Edges are annotated with ``count`` (int), representing the weight.
		- Edges with zero weight are removed after construction.
		- The graph is stored in ``self.g``

		Raises
		------
		ValueError
			If the resulting graph has no edges (i.e., ``df`` is empty or contains only zeros).	
		"""
		
		logger.info("Building graph with %d docs and %d words", df.shape[1], df.shape[0])
		self.g = Graph(directed=False)

		n_docs, n_words = df.shape[1], df.shape[0]
	
		# Add all vertices first
		self.g.add_vertex(n_docs + n_words)
	
		# Create vertex properties
		name = self.g.new_vp("string")
		kind = self.g.new_vp("int")
		self.g.vp["name"] = name
		self.g.vp["kind"] = kind
	
		# Assign doc vertices (loop for names, array for kind)
		for i, doc in enumerate(df.columns):
			name[self.g.vertex(i)] = doc
		kind.get_array()[:n_docs] = 0
	
		# Assign word vertices (loop for names, array for kind)
		for j, word in enumerate(df.index):
			name[self.g.vertex(n_docs + j)] = word
		kind.get_array()[n_docs:] = np.array([get_kind(w) for w in df.index], dtype=int)
	
		# Edge weights
		weight = self.g.new_ep("int")
		self.g.ep["count"] = weight
	
		# Build sparse edges
		rows, cols = df.values.nonzero()
		vals = df.values[rows, cols].astype(int)
		edges = [(c, n_docs + r, v) for r, c, v in zip(rows, cols, vals)]
		if len(edges)==0: raise ValueError("Empty graph")
	
		self.g.add_edge_list(edges, eprops=[weight])
	
		# Remove edges with 0 weight
		filter_edges = self.g.new_edge_property("bool")
		filter_edges.a = weight.a > 0
		self.g.set_edge_filter(filter_edges)
		self.g.purge_edges()
		self.g.clear_filters()
	
		self.documents = df.columns
		self.words = df.index[self.g.vp['kind'].a[n_docs:] == 1]
		for ik in range(2, 2 + self.nbranches):
			self.keywords.append(df.index[self.g.vp['kind'].a[n_docs:] == ik])


	def fit(self, n_init=1, verbose=True, deg_corr=True, overlap=False, parallel=False, B_min=0, B_max=None, clabel=None, *args, **kwargs) -> None:
		"""
		Fit a nested stochastic block self to the graph using `minimize_nested_blockmodel_dl`.
	
		This method performs multiple initializations and keeps the best self 
		based on the minimum description length (entropy). It supports degree-corrected 
		and overlapping block selfs, and can perform parallel moves for efficiency.
	
		Parameters
		----------
		n_init : int, default=1
			Number of random initializations. The self with the lowest entropy is retained.
		verbose : bool, default=True
			If True, print progress messages.
		deg_corr : bool, default=True
			If True, use a degree-corrected block self.
		overlap : bool, default=False
			If True, use an overlapping block self.
		parallel : bool, default=False
			If True, perform parallel moves during optimization.
		B_min : int, default=0
			Minimum number of blocks to consider.
		B_max : int, optional
			Maximum number of blocks to consider. Defaults to the number of vertices.
		clabel : str or property map, optional
			Vertex property to use as initial block assignment. If None, the 'kind' 
			vertex property is used.
		*args : positional arguments
			Additional positional arguments passed to `minimize_nested_blockmodel_dl`.
		**kwargs : keyword arguments
			Additional keyword arguments passed to `minimize_nested_blockmodel_dl`. 
		"""
		if clabel == None:
			clabel = self.g.vp['kind']
			state_args = {'clabel': clabel, 'pclabel': clabel}
		else:
			logger.info("Clabel is %s, assigning partitions to vertices", clabel)
			state_args = {'clabel': clabel, 'pclabel': clabel}
	
		state_args["eweight"] = self.g.ep.count
		min_entropy = np.inf
		best_state = None
		state_args["deg_corr"] = deg_corr
		state_args["overlap"] = overlap

		if B_max is None:
			B_max = self.g.num_vertices()
			
		multilevel_mcmc_args={"B_min": B_min, "B_max": B_max, "verbose": verbose,"parallel" : parallel}

		logger.debug("multilevel_mcmc_args: %s", multilevel_mcmc_args)
		logger.debug("state_args: %s", state_args)

		for i in range(n_init):
			logger.info("Fit number: %d", i)
			state = minimize_nested_blockmodel_dl(self.g, state_args=state_args, multilevel_mcmc_args=multilevel_mcmc_args, *args, **kwargs)
			
			entropy = state.entropy()
			if entropy < min_entropy:
				min_entropy = entropy
				self.state = state
				
		self.mdl = min_entropy

		L = len(self.state.levels)
		self.L = L

		self.groups = {}
		logger.info("Saving data in %s", self.saving_path)
		self.save_data(path_to_save=self.saving_path)

		logger.info("Annotate object")
		self.annotate_obj()


	# Helper functions
	def save_graph(self, filename: str = "graph.xml.gz") -> None:
		"""
		Save the graph

		:param filename: name of the graph stored
		"""
		logger.info("Saving graph to %s", filename)
		self.g.save(filename)
	
	
	def load_graph(self, filename: str = "graph.xml.gz") -> None:
		"""
		Load a saved graph from disk and rebuild documents, words, and keywords.

		Parameters
		----------
		filename : str, optional
			Path to the saved graph file (default: "graph.xml.gz").
		"""
		logger.info("Loading graph from %s", filename)

		self.g = load_graph(filename)
		self.documents = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 0]
		self.words = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 1]
		metadata_indexes = np.unique(self.g.vp["kind"].a)
		metadata_indexes = metadata_indexes[metadata_indexes > 1] #no doc or words
		self.nbranches = len(metadata_indexes)
		for i_keyword in metadata_indexes:
			self.keywords.append([self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == i_keyword])

	
	def _get_edge_cache(self):
		"""
		Cache edge sources, targets, weights, and kinds once per graph.
		"""
		if hasattr(self, "_edge_cache"):
			return self._edge_cache
	
		edge_array = self.g.get_edges()  # shape (m, 2)
		sources = edge_array[:, 0].astype(np.int64)
		targets = edge_array[:, 1].astype(np.int64)
	
		weights = self.g.ep["count"].a.astype(np.float64)
		kinds = self.g.vp["kind"].a[targets].astype(np.int64)
	
		self._edge_cache = {"sources": sources, "targets": targets, "weights": weights, "kinds": kinds}
		return self._edge_cache
	
	
	def _get_state_l_edges_array(self, state_l):
		"""
		Cache block assignments (z1, z2) once per hierarchy level.
		"""
		if hasattr(state_l, "_edges_array_cache"):
			return state_l._edges_array_cache
	
		edges = list(self.g.edges())
		state_l_edges = state_l.get_edge_blocks()
	
		arr = np.empty((len(edges), 2), dtype=np.int64)
		for i, e in enumerate(edges):
			arr[i, 0] = state_l_edges[e][0]
			arr[i, 1] = state_l_edges[e][1]
	
		state_l._edges_array_cache = arr
		return arr
	
	
	def get_groups(self, l=0):
		"""
		Fully optimized get_groups with caching of edges and block assignments.
		No per-edge Python loops in the hot path.
		"""
		if l in self.groups:
			return self.groups[l]
	
	# --- Numba kernel with O(1) offset lookup ---
	@njit
	def process_edges_numba_stack(sources, targets, z1, z2, kinds, weights,
								  D, W, K_arr, offsets, nbranches,
								  n_db, n_wb, n_dbw, n_w_key_b3, n_dbw_key3):
		m = len(sources)
		for i in range(m):
			v1 = sources[i]
			v2 = targets[i]
			w = weights[i]
			t1 = z1[i]
			t2 = z2[i]
			kind = kinds[i]

			# update doc-group counts
			n_db[v1, t1] += w

			if kind == 1:
				# word node
				idx_w = v2 - D
				if 0 <= idx_w < n_wb.shape[0]:
					n_wb[idx_w, t2] += w
				n_dbw[v1, t2] += w

			elif kind >= 2:
				ik = kind - 2
				if 0 <= ik < nbranches:
					idx_k = v2 - offsets[ik]  # O(1) offset lookup
					if 0 <= idx_k < K_arr[ik]:
						n_w_key_b3[ik, idx_k, t2] += w
						n_dbw_key3[ik, v1, t2] += w



		# --- Setup ---
		state_l = self.state.project_level(l).copy(overlap=True)
		B = state_l.get_B()
		D, W, K = self.get_shape()
		nbranches = self.nbranches
		K_arr = np.array(K, dtype=np.int64)
	
		# --- Precompute branch offsets ---
		if nbranches > 0 and K_arr.size > 0:
			prefix_K = np.empty(nbranches, dtype=np.int64)
			prefix_K[0] = 0
			for ii in range(1, nbranches):
				prefix_K[ii] = prefix_K[ii-1] + K_arr[ii-1]
			offsets = (D + W) + prefix_K
		else:
			offsets = np.empty(0, dtype=np.int64)
	
		# --- Get cached edge arrays ---
		edge_cache = self._get_edge_cache()
		sources = edge_cache["sources"]
		targets = edge_cache["targets"]
		weights = edge_cache["weights"]
		kinds = edge_cache["kinds"]
	
		# --- Get cached block assignments ---
		z_pairs = self._get_state_l_edges_array(state_l)
		z1_arr = z_pairs[:, 0]
		z2_arr = z_pairs[:, 1]
	
		# --- Allocate accumulators ---
		n_wb = np.zeros((W, B), dtype=np.float64)
		n_db = np.zeros((D, B), dtype=np.float64)
		n_dbw = np.zeros((D, B), dtype=np.float64)
		n_w_key_b3 = np.zeros((nbranches, np.max(K_arr) if nbranches > 0 else 0, B), dtype=np.float64)
		n_dbw_key3 = np.zeros((nbranches, D, B), dtype=np.float64)
	
		# --- Process edges ---
		process_edges_numba_stack(sources, targets, z1_arr, z2_arr, kinds, weights, D, W, K_arr, offsets, nbranches, n_db, n_wb, n_dbw, n_w_key_b3, n_dbw_key3)
	
		# --- Trim and normalize (unchanged) ---
		ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
		n_db = n_db[:, ind_d]
		Bd = len(ind_d)
	
		ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
		n_wb = n_wb[:, ind_w]
		Bw = len(ind_w)
	
		ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
		n_dbw = n_dbw[:, ind_w2]
	
		n_w_key_b_list, n_dbw_key_list, Bk = [], [], []
		for ik in range(nbranches):
			Kk = int(K_arr[ik]) if K_arr.size > 0 else 0
			if Kk > 0:
				col_sums = np.sum(n_w_key_b3[ik, :Kk, :], axis=0)
				ind_wk = np.where(col_sums > 0)[0]
				if ind_wk.size > 0:
					n_w_key_b_list.append(n_w_key_b3[ik, :Kk, :][:, ind_wk].copy())
				else:
					n_w_key_b_list.append(np.zeros((Kk, 0), dtype=np.float64))
				Bk.append(len(ind_wk))
			else:
				n_w_key_b_list.append(np.zeros((0, 0), dtype=np.float64))
				Bk.append(0)
	
			col_sums_dbw = np.sum(n_dbw_key3[ik], axis=0)
			ind_w2k = np.where(col_sums_dbw > 0)[0]
			if ind_w2k.size > 0:
				n_dbw_key_list.append(n_dbw_key3[ik][:, ind_w2k].copy())
			else:
				n_dbw_key_list.append(np.zeros((D, 0), dtype=np.float64))
	
		# --- Distributions ---
		denom = np.sum(n_wb, axis=1, keepdims=True)
		p_tw_w = (n_wb / denom).T
	
		p_tk_w_key = []
		for ik in range(nbranches):
			arr = n_w_key_b_list[ik]
			denom = np.sum(arr, axis=1, keepdims=True)
			p_tk_w_key.append((arr / denom).T)
	
		denom = np.sum(n_wb, axis=0, keepdims=True)
		p_w_tw = n_wb / denom
	
		p_w_key_tk = []
		for ik in range(nbranches):
			arr = n_w_key_b_list[ik]
			denom = np.sum(arr, axis=0, keepdims=True)
			p_w_key_tk.append(arr / denom)
	
		denom = np.sum(n_dbw, axis=1, keepdims=True)
		p_tw_d = (n_dbw / denom).T
	
		p_tk_d = []
		for ik in range(nbranches):
			arr = n_dbw_key_list[ik]
			denom = np.sum(arr, axis=1, keepdims=True)
			p_tk_d.append((arr / denom).T)
	
		denom = np.sum(n_db, axis=1, keepdims=True)
		p_td_d = (n_db / denom).T
	
		result = {'Bd': Bd, 'Bw': Bw, 'Bk': Bk,
					'p_tw_w': p_tw_w, 'p_tk_w_key': p_tk_w_key, 'p_td_d': p_td_d,
					'p_w_tw': p_w_tw, 'p_w_key_tk': p_w_key_tk, 'p_tw_d': p_tw_d, 'p_tk_d': p_tk_d}
	
		self.groups[l] = result
		return result



	def save_single_level(self, l: int, path_to_save: str) -> None:
		"""
		Save per-level probability matrices (topics, clusters, documents) for the given level.

		Parameters
		----------
		l : int
			The level index to save. Must be within the range of available self levels.
		savingpath_to_save_path : str
			Base path (folder + prefix) where files will be written.
			Example: "results/myself" → files like:
				- results/myself_level_0_mainfeature_topics.tsv.gz
				- results/myself_level_0_clusters.tsv.gz
				- results/myself_level_0_mainfeature_topics_documents.tsv.gz
				- results/myself_level_0_metafeature_topics.tsv.gz
				- results/myself_level_0_metafeature_topics_documents.tsv.gz

		Notes
		-----
		- Files are written as tab-separated values (`.tsv.gz`) with gzip compression.
		- Raises RuntimeError if any file cannot be written.
		"""

		# --- Validate inputs ---
		if not isinstance(l, int) or l < 0 or l >= len(self.state.levels) or l >= len(self.state.levels):
			raise ValueError(f"Invalid level index {l}. Must be between 0 and {len(self.state.levels) - 1}.")
		if not isinstance(path_to_save, str) or not path_to_save.strip():
			raise ValueError("`path_to_save` must be a non-empty string path prefix.")

		main_feature = self.modalities[0]

		try:
			data = self.get_groups(l)
		except Exception as e:
			raise RuntimeError(f"Failed to get group data for level {l}: {e}") from e

		# Helper to safely save a DataFrame
		def _safe_save(df, filepath):
			try:
				Path(filepath).parent.mkdir(parents=True, exist_ok=True)
				df.to_csv(filepath, compression="gzip", sep="\t")
			except Exception as e:
				raise RuntimeError(f"Failed to save {filepath}: {e}") from e

		# --- P(document | cluster) ---
		clusters = pd.DataFrame(data=data["p_td_d"], columns=self.documents)
		_safe_save(clusters, f"{path_to_save}_level_{l}_clusters.tsv.gz")


		# --- P(main_feature | main_topic) ---
		p_w_tw = pd.DataFrame(data=data["p_w_tw"], index=self.words,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		_safe_save(p_w_tw, f"{path_to_save}_level_{l}_{main_feature}_topics.tsv.gz")

		# --- P(main_topic | documents) ---
		p_tw_d = pd.DataFrame(data=data["p_tw_d"].T,index=self.documents,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		_safe_save(p_tw_d, f"{path_to_save}_level_{l}_{main_feature}_topics_documents.tsv.gz")

		# --- P(meta_feature | meta_topic_feature), if any ---
		if len(self.modalities) > 1:
			for k, meta_features in enumerate(self.modalities[1:]):
				p_w_tw = pd.DataFrame(data=data["p_w_key_tk"][k], index=self.keywords[k],
					columns=[f"{meta_features}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(p_w_tw, f"{path_to_save}_level_{l}_{meta_features}_topics.tsv.gz")


			# --- P(meta_topic | document) ---
			for k, meta_features in enumerate(self.modalities[1:]):
				p_tw_d = pd.DataFrame(data=data["p_tk_d"][k].T, index=self.documents,
					columns=[f"{meta_features}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(p_tw_d, f"{path_to_save}_level_{l}_{meta_features}_topics_documents.tsv.gz")



	def save_data(self, path_to_save: str = "results/myself") -> None:
		"""
		Save the global graph, self, state, and level-specific data for the current nSBM self.

		Parameters
		----------
		savinpath_to_saveg_path : str, optional
			Base path (folder + prefix) where all outputs will be saved.
			Example: "results/myself" will produce:
				- results/myself_graph.xml.gz
				- results/myself_self.pkl	
				- results/myself_entropy.txt
				- results/myself_state.pkl
				- results/myself_level_X_*.tsv.gz  (per level, up to 6 levels)

		Notes
		-----
		- The parent folder is created automatically if it does not exist.
		- Level saving is parallelized with threads for efficiency in I/O.
		- By default, at most self.max_depth levels are saved, or fewer if the self has <self.max_depth levels.
		"""
		logger.info("Saving self data to %s", path_to_save)

		L = min(len(self.state.levels), self.max_depth)
		self.L = L
		if L == 0:
			logger.warning("Nothing to save (no levels found)")
			return
		
		folder = os.path.dirname(path_to_save)
		Path(folder).mkdir(parents=True, exist_ok=True)

		try:
			self.save_graph(filename=f"{path_to_save}_graph.xml.gz")
			self.dump_model(filename=f"{path_to_save}_self.pkl")

			with open(f"{path_to_save}_entropy.txt", "w") as f:
				f.write(str(self.state.entropy()))

			with open(f"{path_to_save}_state.pkl", "wb") as f:
				pickle.dump(self.state, f)

		except Exception as e:
			logger.error("Failed to save global files: %s", e)
			raise RuntimeError(f"Failed to save global files for self '{path_to_save}': {e}") from e


		errors = []
		with ThreadPoolExecutor() as executor:
			futures = {executor.submit(self.save_single_level, l, path_to_save): l for l in range(L)}
			for future in as_completed(futures):
				l = futures[future]
				try:
					future.result()
				except Exception as e:
					errors.append((l, str(e)))

		if errors:
			msg = "; ".join([f"Level {l}: {err}" for l, err in errors])
			logger.error("Errors occurred while saving levels: %s", msg)
			raise RuntimeError(f"Errors occurred while saving levels: {msg}")


	def annotate_obj(self) -> None:
		L = min(len(self.state.levels), self.max_depth)
		for l in range(0,L):
			main_feature = self.modalities[0]
			data = self.get_groups(l)
			self.obj.obs[f"Level_{l}_cluster"]=np.argmax(pd.DataFrame(data=data["p_td_d"], columns=self.documents)[self.obj.obs.index], axis=0).astype(str)
			
	
			if isinstance(self.obj, MuData):
				order_var=self.obj[main_feature].var.index
				p_w_tw = pd.DataFrame(data=data["p_w_tw"], index=self.words,
								columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])]).loc[order_var]
				self.obj[main_feature].var[f"Level_{l}_{main_feature}_topic"]=np.argmax(p_w_tw, axis=1).astype(str)

			elif isinstance(self.obj, AnnData):
				order_var=self.obj.var.index			 
				p_w_tw = pd.DataFrame(data=data["p_w_tw"], index=self.words,
								columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])]).loc[order_var]
				self.obj.var[f"Level_{l}_{main_feature}_topic"]=np.argmax(p_w_tw, axis=1).astype(str)

			
			p_tw_d = pd.DataFrame(data=data["p_tw_d"].T,index=self.documents,
					columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])]).loc[self.obj.obs.index]
			p_tw_d=p_tw_d-p_tw_d.mean(axis=0)
			self.obj.obs[f"Level_{l}_{main_feature}"]=np.argmax(p_tw_d, axis=1).astype(str)
		
			if len(self.modalities) > 1:
				for k, meta_feature in enumerate(self.modalities[1:]):
					p_w_tw = pd.DataFrame(data=data["p_w_key_tk"][k], index=self.keywords[k],
						columns=[f"{meta_feature}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
					self.obj[meta_feature].var[f"Level_{l}_{meta_feature}_topic"]=np.argmax(p_w_tw, axis=1).astype(str)
			
				# --- P(meta_topic | document) ---
				for k, meta_feature in enumerate(self.modalities[1:]):
					p_tw_d = pd.DataFrame(data=data["p_tk_d"][k].T, index=self.documents,
						columns=[f"{meta_feature}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
					p_tw_d=p_tw_d-p_tw_d.mean(axis=0)
					self.obj.obs[f"Level_{l}_{meta_feature}"]=np.argmax(p_tw_d, axis=1).astype(str)

	def dump_model(self, filename="bionsbm.pkl"):
		"""
		Dump self using pickle

		"""
		logger.info("Dumping self to %s", filename)

		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def load_model(self, filename="bionsbm.pkl"):
		logger.info("Loading self from %s", filename)

		with open(filename, "rb") as f:
			self = pickle.load(f)
		return self

	def get_V(self):
		'''
		return number of word-nodes == types
		'''
		return int(np.sum(self.g.vp['kind'].a == 1))  # no. of types

	def get_D(self):
		'''
		return number of doc-nodes == number of documents
		'''
		return int(np.sum(self.g.vp['kind'].a == 0))  # no. of types

	def get_N(self):
		'''
		return number of edges == tokens
		'''
		return int(self.g.num_edges())  # no. of types


	def get_mdl(self):
		"""
		Get minimum description length

		Proxy to self.state.entropy()
		"""
		return self.mdl
			
	def get_shape(self):
		"""
		:return: list of tuples (number of documents, number of words, (number of keywords,...))
		"""
		D = int(np.sum(self.g.vp['kind'].a == 0)) #documents
		W = int(np.sum(self.g.vp['kind'].a == 1)) #words
		K = [int(np.sum(self.g.vp['kind'].a == (k+2))) for k in range(self.nbranches)] #keywords
		return D, W, K

##### Drawing
	def draw(self, *args, **kwargs) -> None:
		"""
		Draw the network

		:param \*args: positional arguments to pass to self.state.draw
		:param \*\*kwargs: keyword argument to pass to self.state.draw
		"""
		colmap = self.g.vertex_properties["color"] = self.g.new_vertex_property(
			"vector<double>")
		#https://medialab.github.io/iwanthue/
		colors = [  [174,80,209],
					[108,192,70],
					[207, 170, 60],
					[131,120,197],
					[126,138,65],
					[201,90,138],
					[87,172,125],
					[213,73,57],
					[85,175,209],
					[193,120,81]]
		for v in self.g.vertices():
			k = self.g.vertex_properties['kind'][v]
			if k < 10:
				color = np.array(colors[k])/255.
			else:
				color = np.array([187, 129, 164])/255.
			colmap[v] = color
		self.state.draw(
			subsample_edges = 5000, 
			edge_pen_width = self.g.ep["count"],
			vertex_color=colmap,
			vertex_fill_color=colmap, *args, **kwargs)
