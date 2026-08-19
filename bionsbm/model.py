"""
bionSBM 19/08/2026 -- new make_graph, get_groups, save_levels

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
@njit(cache=True, nogil=True)
def _accumulate_doc_blocks(sources, z, weights, out):
	for i in range(len(sources)):
		out[sources[i], z[i]] += weights[i]


@njit(cache=True, nogil=True)
def _accumulate_feature_blocks(sources, targets, z, weights, out_feature, out_doc):
	for i in range(len(sources)):
		w = weights[i]
		out_feature[targets[i], z[i]] += w
		out_doc[sources[i], z[i]] += w


######################################
class bionsbm():
	"""
	Class to run bionsbm
	"""
	def __init__(self, obj, label: Optional[str] = None, max_depth: int = 6, saving_path: str = "results/myself", 
				load_graph_path=None, save_graph_path=None, annotate_input_object=False):
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
			Base path for saving model outputs (graph, state, results).
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
		self.modalities: List[str] = list(obj.mod.keys()) if isinstance(obj, MuData) else [obj.uns["Mod"]]
		self.max_depth: int = max_depth
		self.obj: Any = obj
		self.saving_path = saving_path
		self.load_graph_path = load_graph_path
		self.save_graph_path = save_graph_path
		self.annotate_input_object = annotate_input_object

		if load_graph_path is not None:
			logger.info(f"Loading graph from {load_graph_path}")
			self.load_graph(filename=load_graph_path)
		else:
			self.make_graph(obj)

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
			del g_raw
		else:
			node_type=None
		self.node_type=node_type 

		

	def make_graph(self, obj: Optional[Any] = None) -> None:
		"""
		Build the heterogeneous graph directly from AnnData/MuData matrices.

		Sparse inputs are processed from ``.X`` without conversion to dense pandas
		DataFrames. Documents occupy the first vertex block and each modality gets
		a contiguous feature block. Only positive integer-weight edges are inserted.
		"""
		obj = self.obj if obj is None else obj
		if isinstance(obj, MuData):
			adatas = [obj[key] for key in self.modalities]
		elif isinstance(obj, AnnData):
			adatas = [obj]
		else:
			raise TypeError("make_graph expects an AnnData or MuData object")

		documents = pd.Index(adatas[0].obs_names)
		D = len(documents)
		sizes = np.array([adata.n_vars for adata in adatas], dtype=np.int64)
		offsets = D + np.r_[0, np.cumsum(sizes[:-1])]
		self.nbranches = len(adatas) - 1

		logger.info("Building sparse graph with %d docs and %d feature branches", D, len(adatas))
		self.g = Graph(directed=False)
		self.g.add_vertex(int(D + sizes.sum()))

		name = self.g.vp["name"] = self.g.new_vp("string")
		kind = self.g.vp["kind"] = self.g.new_vp("int")
		weight = self.g.ep["count"] = self.g.new_ep("int")
		kind.a[:D] = 0

		for i, doc in enumerate(documents):
			name[self.g.vertex(i)] = str(doc)

		self.words = pd.Index(adatas[0].var_names.copy())
		self.keywords = []
		for i, (adata, offset) in enumerate(zip(adatas, offsets)):
			prefix = "#" * i
			feature_names = pd.Index([prefix + str(v) for v in adata.var_names]) if i else self.words
			kind.a[offset:offset + adata.n_vars] = i + 1
			for j, feature in enumerate(feature_names):
				name[self.g.vertex(int(offset + j))] = str(feature)
			if i:
				self.keywords.append(feature_names)

		total_edges = 0
		for adata, offset in zip(adatas, offsets):
			X = adata.X.to_memory() if hasattr(adata.X, "to_memory") else adata.X
			if sparse.issparse(X):
				X = X.tocsr(copy=False)
				if not X.has_canonical_format:
					X = X.copy()
					X.sum_duplicates()
				coo = X.tocoo(copy=False)
				rows, cols, vals = coo.row, coo.col, coo.data.astype(np.int64, copy=False)
			else:
				X = np.asarray(X)
				rows, cols = np.nonzero(X)
				vals = X[rows, cols].astype(np.int64, copy=False)

			if adata.obs_names.equals(documents):
				sources = rows.astype(np.int64, copy=False)
			else:
				row_map = documents.get_indexer(adata.obs_names)
				sources = row_map[rows]

			keep = (sources >= 0) & (vals > 0)
			n = int(np.count_nonzero(keep))
			if not n:
				continue

			edges = np.empty((n, 3), dtype=np.int64)
			edges[:, 0] = sources[keep]
			edges[:, 1] = int(offset) + cols[keep]
			edges[:, 2] = vals[keep]
			self.g.add_edge_list(edges, eprops=[weight])
			total_edges += n

		if total_edges == 0:
			raise ValueError("Empty graph")

		self.documents = documents
		if self.save_graph_path is not None:
			folder = os.path.dirname(self.save_graph_path)
			if folder:
				Path(folder).mkdir(parents=True, exist_ok=True)
			self.save_graph(filename=self.save_graph_path)


	def fit(self, n_init=1, verbose=True, deg_corr=True, overlap=False, parallel=False, B_min=0, B_max=None, clabel=None, save=True, *args, **kwargs) -> None:
		"""
		Fit a nested stochastic block model to the graph using `minimize_nested_blockmodel_dl`.
	
		This method performs multiple initializations and keeps the best model 
		based on the minimum description length (entropy). It supports degree-corrected 
		and overlapping block selfs, and can perform parallel moves for efficiency.
	
		Parameters
		----------
		n_init : int, default=1
			Number of random initializations. The model with the lowest entropy is retained.
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
		if save:
			logger.info("Saving data in %s", self.saving_path)
			self.save_data()

		if self.annotate_input_object:
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

		self.g = load_graph(filename)
		self.documents = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 0]
		self.words = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 1]
		metadata_indexes = np.unique(self.g.vp["kind"].a)
		metadata_indexes = metadata_indexes[metadata_indexes > 1] #no doc or words
		self.nbranches = len(metadata_indexes)
		for i_keyword in metadata_indexes:
			self.keywords.append([self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == i_keyword])

	
	def _get_edge_cache(self):
		"""Cache edge arrays in one consistent graph-tool traversal order."""
		if hasattr(self, "_edge_cache"):
			return self._edge_cache

		edges = self.g.get_edges([self.g.edge_index, self.g.ep["count"]])
		sources = edges[:, 0].astype(np.int64, copy=False)
		targets = edges[:, 1].astype(np.int64, copy=False)
		edge_index = edges[:, 2].astype(np.int64, copy=False)
		weights = edges[:, 3].astype(np.float64, copy=False)
		kinds = self.g.vp["kind"].a[targets].astype(np.int64, copy=False)
		branch_edges = [np.flatnonzero(kinds == kind) for kind in range(1, self.nbranches + 2)]

		self._edge_cache = {
			"sources": sources, "targets": targets, "edge_index": edge_index,
			"weights": weights, "kinds": kinds, "branch_edges": branch_edges
		}
		return self._edge_cache


	def _get_state_l_edges_array(self, state_l, edge_index):
		"""Return edge-end block assignments aligned to ``Graph.get_edges()`` order."""
		z = state_l.get_edge_blocks().get_2d_array([0, 1]).T
		return z[edge_index].astype(np.int64, copy=False)


	def get_groups(self, l=0, cache=True):
		"""Compute compact per-level distributions without global padded block tensors."""
		if cache and l in self.groups:
			return self.groups[l]

		state_l = self.state.project_level(l).copy(overlap=True)
		D, W, K = self.get_shape()
		K_arr = np.asarray(K, dtype=np.int64)
		edge_cache = self._get_edge_cache()
		sources, targets, weights = edge_cache["sources"], edge_cache["targets"], edge_cache["weights"]
		z = self._get_state_l_edges_array(state_l, edge_cache["edge_index"])

		# Document blocks: all edges contribute.
		_, z_doc = np.unique(z[:, 0], return_inverse=True)
		z_doc = z_doc.astype(np.int64, copy=False)
		n_db = np.zeros((D, int(z_doc.max()) + 1 if z_doc.size else 0), dtype=np.float64)
		_accumulate_doc_blocks(sources, z_doc, weights, n_db)
		Bd = n_db.shape[1]

		# Main feature branch.
		main_idx = edge_cache["branch_edges"][0]
		if main_idx.size:
			_, z_main = np.unique(z[main_idx, 1], return_inverse=True)
			z_main = z_main.astype(np.int64, copy=False)
			n_wb = np.zeros((W, int(z_main.max()) + 1), dtype=np.float64)
			n_dbw = np.zeros((D, n_wb.shape[1]), dtype=np.float64)
			_accumulate_feature_blocks(
				sources[main_idx], targets[main_idx] - D, z_main, weights[main_idx], n_wb, n_dbw
			)
		else:
			n_wb = np.zeros((W, 0), dtype=np.float64)
			n_dbw = np.zeros((D, 0), dtype=np.float64)
		Bw = n_wb.shape[1]

		# Metadata branches are processed one at a time instead of allocating padded 3-D tensors.
		n_w_key_b_list, n_dbw_key_list, Bk = [], [], []
		offset = D + W
		for ik, Kk in enumerate(K_arr):
			idx = edge_cache["branch_edges"][ik + 1]
			if idx.size:
				_, z_key = np.unique(z[idx, 1], return_inverse=True)
				z_key = z_key.astype(np.int64, copy=False)
				n_key = np.zeros((int(Kk), int(z_key.max()) + 1), dtype=np.float64)
				n_doc_key = np.zeros((D, n_key.shape[1]), dtype=np.float64)
				_accumulate_feature_blocks(
					sources[idx], targets[idx] - offset, z_key, weights[idx], n_key, n_doc_key
				)
			else:
				n_key = np.zeros((int(Kk), 0), dtype=np.float64)
				n_doc_key = np.zeros((D, 0), dtype=np.float64)
			n_w_key_b_list.append(n_key)
			n_dbw_key_list.append(n_doc_key)
			Bk.append(n_key.shape[1])
			offset += int(Kk)

		# Preserve the established normalization order for stable serialized values.
		denom = np.sum(n_wb, axis=1, keepdims=True)
		p_tw_w = (n_wb / denom).T

		p_tk_w_key = []
		for arr in n_w_key_b_list:
			denom = np.sum(arr, axis=1, keepdims=True)
			p_tk_w_key.append((arr / denom).T)

		denom = np.sum(n_wb, axis=0, keepdims=True)
		p_w_tw = n_wb / denom

		p_w_key_tk = []
		for arr in n_w_key_b_list:
			denom = np.sum(arr, axis=0, keepdims=True)
			p_w_key_tk.append(arr / denom)

		denom = np.sum(n_dbw, axis=1, keepdims=True)
		p_tw_d = (n_dbw / denom).T

		p_tk_d = []
		for arr in n_dbw_key_list:
			denom = np.sum(arr, axis=1, keepdims=True)
			p_tk_d.append((arr / denom).T)

		denom = np.sum(n_db, axis=1, keepdims=True)
		p_td_d = (n_db / denom).T

		result = {
			'Bd': Bd, 'Bw': Bw, 'Bk': Bk,
			'p_tw_w': p_tw_w, 'p_tk_w_key': p_tk_w_key, 'p_td_d': p_td_d,
			'p_w_tw': p_w_tw, 'p_w_key_tk': p_w_key_tk, 'p_tw_d': p_tw_d, 'p_tk_d': p_tk_d
		}
		if cache:
			self.groups[l] = result
		return result


	def save_single_level(self, l: int) -> None:
		"""Compute and save all probability tables for one hierarchy level."""
		if not isinstance(l, int) or l < 0 or l >= len(self.state.levels):
			raise ValueError(f"Invalid level index {l}. Must be between 0 and {len(self.state.levels) - 1}.")
		if not isinstance(self.saving_path, str) or not self.saving_path.strip():
			raise ValueError("`self.saving_path` must be a non-empty string path prefix.")

		# Do not retain large level matrices unless annotate_obj() will reuse them.
		try:
			data = self.get_groups(l, cache=self.annotate_input_object)
		except Exception as e:
			raise RuntimeError(f"Failed to get group data for level {l}: {e}") from e

		main_feature = self.modalities[0]

		def save_tsv(df, suffix):
			filepath = f"{self.saving_path}_level_{l}_{suffix}.tsv.gz"
			try:
				Path(filepath).parent.mkdir(parents=True, exist_ok=True)
				df.to_csv(filepath, compression="gzip", sep="\t")
			except Exception as e:
				raise RuntimeError(f"Failed to save {filepath}: {e}") from e

		save_tsv(pd.DataFrame(data["p_td_d"], columns=self.documents), "clusters")

		columns = [f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])]
		save_tsv(pd.DataFrame(data["p_w_tw"], index=self.words, columns=columns), f"{main_feature}_topics")
		save_tsv(pd.DataFrame(data["p_tw_d"].T, index=self.documents, columns=columns),
				 f"{main_feature}_topics_documents")

		for k, meta_feature in enumerate(self.modalities[1:]):
			columns = [f"{meta_feature}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])]
			save_tsv(pd.DataFrame(data["p_w_key_tk"][k], index=self.keywords[k], columns=columns),
					 f"{meta_feature}_topics")

			# Preserve the historical plural "topics" in document-topic column names.
			doc_columns = [f"{meta_feature}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])]
			save_tsv(pd.DataFrame(data["p_tk_d"][k].T, index=self.documents, columns=doc_columns),
					 f"{meta_feature}_topics_documents")


	def save_levels(self, max_workers=None) -> None:
		"""Save hierarchy levels in parallel while sharing the immutable edge cache."""
		L = min(len(self.state.levels), self.max_depth)
		if L == 0:
			return

		# Build once before worker threads start, avoiding lazy-initialization races.
		self._get_edge_cache()

		errors = []
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = {executor.submit(self.save_single_level, l): l for l in range(L)}
			for future in as_completed(futures):
				l = futures[future]
				try:
					future.result()
				except Exception as e:
					errors.append((l, str(e)))

		if errors:
			msg = "; ".join(f"Level {l}: {err}" for l, err in errors)
			logger.error("Errors occurred while saving levels: %s", msg)
			raise RuntimeError(f"Errors occurred while saving levels: {msg}")


	def save_data(self, max_workers=None) -> None:
		"""
		Save the graph, model, state, entropy, and level-specific bionSBM outputs.

		Parameters
		----------
		self.saving_path : str
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
		- By default, at most self.max_depth levels are saved, or fewer if the model has <self.max_depth levels.
		"""
		logger.info("Saving model data to %s", self.saving_path)

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
			raise RuntimeError(f"Failed to save global files for model '{self.saving_path}': {e}") from e


		# Keep parallelization across hierarchy levels.
		self.save_levels(max_workers=max_workers)


	def annotate_obj(self) -> None:
		L = min(len(self.state.levels), self.max_depth)
		for l in range(0, L):
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
		Dump model using pickle

		"""
		logger.info("Dumping model to %s", filename)

		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def load_model(self, filename="bionsbm.pkl"):
		logger.info("Loading model from %s", filename)

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
