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

		
	def make_graph(self, df_one: pd.DataFrame, df_keyword_list: List[pd.DataFrame]) -> None:
		df_all = df_one.copy(deep =True)
		for ikey,df_keyword in enumerate(df_keyword_list):
			df_keyword = df_keyword.reindex(columns=df_one.columns)
			df_keyword.index = ["".join(["#" for _ in range(ikey+1)])+str(keyword) for keyword in df_keyword.index]
			df_keyword["kind"] = ikey+2
			df_all = pd.concat((df_all, df_keyword), axis=0)
		del df_keyword, df_one

		kinds=pd.DataFrame(df_all["kind"].fillna(1)) if len()

		self.nbranches = len(df_keyword_list)
		del df_keyword_list
			
		df_all.drop("kind", axis=1, errors='ignore', inplace=True)
		
		self.g = Graph(directed=False)

		n_docs, n_words = df_all.shape[1], df_all.shape[0]

		# Add all vertices first
		self.g.add_vertex(n_docs + n_words)

		# Create vertex properties
		name = self.g.new_vp("string")
		kind = self.g.new_vp("int")
		self.g.vp["name"] = name
		self.g.vp["kind"] = kind

		# Assign doc vertices (loop for names, array for kind)
		for i, doc in enumerate(df_all.columns):
			name[self.g.vertex(i)] = doc
		kind.get_array()[:n_docs] = 0

		# Assign word vertices (loop for names, array for kind)
		for j, word in enumerate(df_all.index):
			name[self.g.vertex(n_docs + j)] = word
		kind.get_array()[n_docs:] = np.array([int(kinds.at[w,"kind"]) for w in df_all.index], dtype=int)

		# Edge weights
		weight = self.g.new_ep("int")
		self.g.ep["count"] = weight

		# Build sparse edges
		rows, cols = df_all.values.nonzero()
		vals = df_all.values[rows, cols].astype(int)
		edges = [(c, n_docs + r, v) for r, c, v in zip(rows, cols, vals)]
		if len(edges)==0: raise ValueError("Empty graph")

		self.g.add_edge_list(edges, eprops=[weight])

		# Remove edges with 0 weight
		filter_edges = self.g.new_edge_property("bool")
		filter_edges.a = weight.a > 0
		self.g.set_edge_filter(filter_edges)
		self.g.purge_edges()
		self.g.clear_filters()

		self.documents = df_all.columns
		self.words = df_all.index[self.g.vp['kind'].a[n_docs:] == 1]
		for ik in range(2, 2 + self.nbranches):
			self.keywords.append(df_all.index[self.g.vp['kind'].a[n_docs:] == ik])


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
		self.save_data()

		logger.info("Annotate object")
		self.annotate_obj()

	def get_groups(self, l=0):
		"""
		Compute group-level summary matrices

		Parameters
		----------
		l : int, default=0
			Hierarchical level to project the fitted nested blockmodel to.

		Returns
		-------
		dict
			A dictionary with the following keys (matching the original `get_groups`):
			- 'Bd' : int
				Number of active document groups (after pruning empty columns).
			- 'Bw' : int
				Number of active word groups (after pruning).
			- 'Bk' : list[int]
				Number of active keyword groups per keyword branch (after pruning).
			- 'p_tw_w' : np.ndarray, shape (Bw, W)
				Group membership of each word node: P(t_w | w). Rows sum to 1; rows
				corresponding to words with zero mass are all-NaN.
			- "p_tk_w_key" : list[np.ndarray]
				For each keyword branch `ik`, matrix of shape (Bk[ik], K[ik]) with
				P(t_k | keyword). Rows with zero mass are all-NaN.
			- 'p_td_d' : np.ndarray, shape (Bd, D)
				Group membership of each document node: P(t_d | d). Rows with zero
				mass are all-NaN.
			- 'p_w_tw' : np.ndarray, shape (W, Bw)
				Topic distribution for words: P(w | t_w). Columns with zero mass
				are all-NaN.
			- 'p_w_key_tk' : list[np.ndarray]
				For each keyword branch `ik`, matrix of shape (K[ik], Bk[ik]) with
				P(keyword | t_k). Columns with zero mass are all-NaN.
			- 'p_tw_d' : np.ndarray, shape (Bw, D)
				Mixture of word-groups in documents: P(t_w | d). Rows with zero
				mass are all-NaN.
			- 'p_tk_d' : list[np.ndarray]
				For each keyword branch `ik`, matrix of shape (Bk[ik], D) with
				P(t_k | d). Rows with zero mass are all-NaN.

		"""


		if l in self.groups.keys():
			return self.groups[l]
		# --- project to level; non-overlap for speed & simple b[v] array ---
		state_l = self.state.project_level(l).copy(overlap=False)
		b_arr = state_l.get_blocks().a.astype(np.int64)
		B = int(state_l.get_B())
	
		# --- basic shapes ---
		D, W, K = self.get_shape()
		K = list(K)
		K_cumsum = np.cumsum([0] + K)				  # [0, K0, K0+K1, ...]
		KW_offsets = D + W + K_cumsum[:-1]			 # global start index for each keyword branch
	
		# --- pull edges in bulk: src, tgt, weight ---
		e_mat = self.g.get_edges(eprops=[self.g.ep["count"]])
		src = e_mat[:, 0].astype(np.int64)
		tgt = e_mat[:, 1].astype(np.int64)
		w   = e_mat[:, 2].astype(np.int64)
	
		z_src = b_arr[src]
		z_tgt = b_arr[tgt]
	
		kind = self.g.vp['kind'].a
		kind_tgt = kind[tgt]
	
		# --- alloc accumulators ---
		n_wb	  = np.zeros((W, B), dtype=np.int64)
		n_db	  = np.zeros((D, B), dtype=np.int64)
		n_dbw	 = np.zeros((D, B), dtype=np.int64)
		n_w_key_b = [np.zeros((K[ik], B), dtype=np.int64) for ik in range(self.nbranches)]
		n_dbw_key = [np.zeros((D, B), dtype=np.int64)	 for _  in range(self.nbranches)]
	
		# --- accumulate (vectorized) ---
		# docs are sources by construction
		np.add.at(n_db, (src, z_src), w)
	
		# words as targets (kind == 1)
		mask_w = (kind_tgt == 1)
		if mask_w.any():
			w_idx = tgt[mask_w] - D
			np.add.at(n_wb,  (w_idx,		   z_tgt[mask_w]), w[mask_w])
			np.add.at(n_dbw, (src[mask_w],	 z_tgt[mask_w]), w[mask_w])
	
		# keywords as targets (kind >= 2)
		if self.nbranches > 0:
			mask_kw = (kind_tgt >= 2)
			if mask_kw.any():
				kw_kinds = kind_tgt[mask_kw]  # values in {2,3,...}
				sel_kw = np.where(mask_kw)[0]
				for ik in range(self.nbranches):
					m = (kw_kinds == (ik + 2))
					if not m.any():
						continue
					sel = sel_kw[m]
					kw_local = tgt[sel] - KW_offsets[ik]
					np.add.at(n_w_key_b[ik], (kw_local,   z_tgt[sel]), w[sel])
					np.add.at(n_dbw_key[ik], (src[sel],   z_tgt[sel]), w[sel])
	
		# --- prune empty columns exactly like original ---
		ind_d = np.where(np.sum(n_db,  axis=0) > 0)[0];  Bd = len(ind_d);  n_db  = n_db[:,  ind_d]
		ind_w = np.where(np.sum(n_wb,  axis=0) > 0)[0];  Bw = len(ind_w);  n_wb  = n_wb[:,  ind_w]
		ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0];					   n_dbw = n_dbw[:, ind_w2]
	
		ind_w_key, ind_w2_keyword, Bk = [], [], []
		for ik in range(self.nbranches):
			idx1 = np.where(np.sum(n_w_key_b[ik],  axis=0) > 0)[0]
			idx2 = np.where(np.sum(n_dbw_key[ik],  axis=0) > 0)[0]
			ind_w_key.append(idx1); ind_w2_keyword.append(idx2); Bk.append(len(idx1))
			n_w_key_b[ik] = n_w_key_b[ik][:, idx1]
			n_dbw_key[ik] = n_dbw_key[ik][:, idx2]
	
		# --- NaN-preserving normalizers (match original semantics) ---
		def _row_norm_nan(M: np.ndarray) -> np.ndarray:
			M = M.astype(float, copy=False)
			s = M.sum(axis=1, keepdims=True)
			out = np.full_like(M, np.nan, dtype=float)
			valid = (s[:, 0] != 0)
			if np.any(valid):
				out[valid] = M[valid] / s[valid]
			return out.T  # original returns transposed
	
		def _col_norm_nan(M: np.ndarray) -> np.ndarray:
			M = M.astype(float, copy=False)
			s = M.sum(axis=0, keepdims=True)
			out = np.full_like(M, np.nan, dtype=float)
			valid = (s[0] != 0)
			if np.any(valid):
				out[:, valid] = M[:, valid] / s[:, valid]
			return out
	
		# --- probabilities (identical layout to original) ---
		p_tw_w	  = _row_norm_nan(n_wb)
		p_tk_w_key  = [_row_norm_nan(n_w_key_b[ik]) for ik in range(self.nbranches)]
		p_w_tw	  = _col_norm_nan(n_wb)
		p_w_key_tk  = [_col_norm_nan(n_w_key_b[ik]) for ik in range(self.nbranches)]
		p_tw_d	  = _row_norm_nan(n_dbw)
		p_tk_d	  = [_row_norm_nan(n_dbw_key[ik]) for ik in range(self.nbranches)]
		p_td_d	  = _row_norm_nan(n_db)
	
		result = dict(
			Bd=Bd, Bw=Bw, Bk=Bk,
			p_tw_w=p_tw_w, p_tk_w_key=p_tk_w_key, p_td_d=p_td_d,
			p_w_tw=p_w_tw, p_w_key_tk=p_w_key_tk, p_tw_d=p_tw_d, p_tk_d=p_tk_d
		)
		self.groups[l] = result
		return result

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


	def save_single_level(self, l: int) -> None:
		"""
		Save per-level probability matrices (topics, clusters, documents) for the given level.

		Parameters
		----------
		l : int
			The level index to save. Must be within the range of available self levels.
		saving in self.saving_path_path : str
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
		if not isinstance(self.saving_path, str) or not self.saving_path.strip():
			raise ValueError("`self.saving_path` must be a non-empty string path prefix.")

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
		_safe_save(clusters, f"{self.saving_path}_level_{l}_clusters.tsv.gz")


		# --- P(main_feature | main_topic) ---
		p_w_tw = pd.DataFrame(data=data["p_w_tw"], index=self.words,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		_safe_save(p_w_tw, f"{self.saving_path}_level_{l}_{main_feature}_topics.tsv.gz")

		# --- P(main_topic | documents) ---
		p_tw_d = pd.DataFrame(data=data["p_tw_d"].T,index=self.documents,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		_safe_save(p_tw_d, f"{self.saving_path}_level_{l}_{main_feature}_topics_documents.tsv.gz")

		# --- P(meta_feature | meta_topic_feature), if any ---
		if len(self.modalities) > 1:
			for k, meta_features in enumerate(self.modalities[1:]):
				p_w_tw = pd.DataFrame(data=data["p_w_key_tk"][k], index=self.keywords[k],
					columns=[f"{meta_features}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(p_w_tw, f"{self.saving_path}_level_{l}_{meta_features}_topics.tsv.gz")


			# --- P(meta_topic | document) ---
			for k, meta_features in enumerate(self.modalities[1:]):
				p_tw_d = pd.DataFrame(data=data["p_tk_d"][k].T, index=self.documents,
					columns=[f"{meta_features}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(p_tw_d, f"{self.saving_path}_level_{l}_{meta_features}_topics_documents.tsv.gz")


	def save_data(self) -> None:
		"""
		Save the global graph, self, state, and level-specific data for the current nSBM self.

		Parameters
		----------
		saving in self.saving_pathg_path : str, optional
			Base path (folder + prefix) where all outputs will be saved.
			Example: "results/myself" will produce:
				- results/myself_graph.xml.gz
				- results/myself_model.pkl	
				- results/myself_entropy.txt
				- results/myself_state.pkl
				- results/myself_level_X_*.tsv.gz  (per level, up to 6 levels)

		Notes
		-----
		- The parent folder is created automatically if it does not exist.
		- Level saving is parallelized with threads for efficiency in I/O.
		- By default, at most self.max_depth levels are saved, or fewer if the self has <self.max_depth levels.
		"""
		logger.info("Saving self data to %s", self.saving_path)

		L = min(len(self.state.levels), self.max_depth)
		self.L = L
		if L == 0:
			logger.warning("Nothing to save (no levels found)")
			return
		
		folder = os.path.dirname(self.saving_path)
		Path(folder).mkdir(parents=True, exist_ok=True)

		try:
			self.save_graph(filename=f"{self.saving_path}_graph.xml.gz")
			self.dump_model(filename=f"{self.saving_path}_model.pkl")

			with open(f"{self.saving_path}_entropy.txt", "w") as f:
				f.write(str(self.state.entropy()))

			with open(f"{self.saving_path}_state.pkl", "wb") as f:
				pickle.dump(self.state, f)

		except Exception as e:
			logger.error("Failed to save global files: %s", e)
			raise RuntimeError(f"Failed to save global files for self '{self.saving_path}': {e}") from e


		errors = []
		with ThreadPoolExecutor() as executor:
			futures = {executor.submit(self.save_single_level, l): l for l in range(L)}
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
