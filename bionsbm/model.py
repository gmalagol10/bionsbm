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

import graph_tool.all as gt
import numpy as np
import pandas as pd
import cloudpickle as pickle
import scanpy as sc
import anndata as ad
import muon as mu

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import sparse
from numba import njit

from . import sbmtm


"""
Inherit hSBM code from https://github.com/martingerlach/hSBM_Topicmodel
"""
from . import sbmtm


class bionsbm(sbmtm.sbmtm):
	"""
	Class to run bionsbm
	"""
	def __init__(self, obj, label=None, max_depth=6):
		super().__init__()
		self.keywords = []
		self.nbranches = 1
		self.modalities = []
		self.max_depth = max_depth

		if isinstance(obj, mu.MuData):
			self.modalities=list(obj.mod.keys())   
			dfs=[obj[key].to_df().T for key in self.modalities]
			self.make_graph_multiple_df(dfs[0], dfs[1:])

		elif isinstance(obj, ad.AnnData):
			self.modalities=["Mod1"]
			self.make_graph_multiple_df(obj.to_df().T)

		if label:
			g_raw=self.g.copy()
			print("Label found")
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


		
	def save_graph(self, filename="graph.xml.gz")->None:
		"""
		Save the graph

		:param filename: name of the graph stored
		"""
		self.g.save(filename)
		
	def load_graph(self, filename="graph.xml.gz")->None:
		"""
		Load a saved graph from disk and rebuild documents, words, and keywords.

		Parameters
		----------
		filename : str, optional
		    Path to the saved graph file (default: "graph.xml.gz").
		"""

		self.g = gt.load_graph(filename)
		self.documents = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 0]
		self.words = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 1]
		metadata_indexes = np.unique(self.g.vp["kind"].a)
		metadata_indexes = metadata_indexes[metadata_indexes > 1] #no doc or words
		self.nbranches = len(metadata_indexes)
		for i_keyword in metadata_indexes:
			self.keywords.append([self.g.vp['name'][v]
										for v in self.g.vertices() if self.g.vp['kind'][v] == i_keyword])


	def make_graph_multiple_df(self, df: pd.DataFrame, df_keyword_list: list)->None:
		"""
		Create a graph from two dataframes one with words, others with keywords or other layers of information

		:param df: DataFrame with words on index and texts on columns
		:param df_keyword_list: list of DataFrames with keywords on index and texts on columns
		"""
		df_all = df.copy(deep =True)
		for ikey,df_keyword in enumerate(df_keyword_list):
			df_keyword = df_keyword.reindex(columns=df.columns)
			df_keyword.index = ["".join(["#" for _ in range(ikey+1)])+str(keyword) for keyword in df_keyword.index]
			df_keyword["kind"] = ikey+2
			df_all = pd.concat((df_all,df_keyword), axis=0)

		def get_kind(word):
			return 1 if word in df.index else df_all.at[word,"kind"]

		self.nbranches = len(df_keyword_list)
	   
		self.make_graph(df_all.drop("kind", axis=1), get_kind)


	def make_graph(self, df: pd.DataFrame, get_kind):
		self.g = gt.Graph(directed=False)

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
		for e in self.g.edges():
		    filter_edges[e] = weight[e] > 0
		self.g.set_edge_filter(filter_edges)
		self.g.purge_edges()
		self.g.clear_filters()

		self.documents = df.columns
		self.words = df.index[self.g.vp['kind'].a[n_docs:] == 1]
		for ik in range(2, 2 + self.nbranches):
		    self.keywords.append(df.index[self.g.vp['kind'].a[n_docs:] == ik])



	def fit(self, n_init=1, verbose=True, deg_corr=True, overlap=False, parallel=False, B_min=0, B_max=None, clabel=None, *args, **kwargs) -> None:
		"""
		Fit using minimize_nested_blockmodel_dl
		
		:param n_init: number of initialisation. The best will be kept
		:param verbose: Print output
		:param deg_corr: use deg corrected model
		:param overlap: use overlapping model
		:param parallel: perform parallel moves
		:param  \*args: positional arguments to pass to gt.minimize_nested_blockmodel_dl
		:param  \*\*kwargs: keywords arguments to pass to gt.minimize_nested_blockmodel_dl
		"""
		if clabel == None:
			clabel = self.g.vp['kind']
			state_args = {'clabel': clabel, 'pclabel': clabel}
		else:
			print(f"Clabel is {clabel}, assigning partitions to vertices", flush=True)
			state_args = {'clabel': clabel, 'pclabel': clabel}
	
		state_args["eweight"] = self.g.ep.count
		min_entropy = np.inf
		best_state = None
		state_args["deg_corr"] = deg_corr
		state_args["overlap"] = overlap

		if B_max is None:
			B_max = self.g.num_vertices()

		print("multilevel_mcmc_args is \n", multilevel_mcmc_args, flush=True)
		print("state_args is \n", state_args, flush=True)

		for _ in range(n_init):
			print("Fit number:", _, flush=True)
			state = gt.minimize_nested_blockmodel_dl(self.g, state_args=state_args,
													multilevel_mcmc_args={"B_min": B_min, "B_max": B_max, "verbose": verbose,"parallel" : parallel}, 
													*args, **kwargs)
			
			entropy = state.entropy()
			if entropy < min_entropy:
				min_entropy = entropy
				self.state = state
				
		self.mdl = min_entropy

		L = len(self.state.levels)
		self.L = L
		self.groups = {}


	def dump_model(self, filename="bionsbm.pkl"):
		"""
		Dump model using pickle

		"""
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def load_model(self, filename="bionsbm.pkl"):
		with open(filename, "rb") as f:
			model = pickle.load(f)
		return model


	def get_mdl(self):
		"""
		Get minimum description length

		Proxy to self.state.entropy()
		"""
		return super().get_mdl()
			
	def _get_shape(self):
		"""
		:return: list of tuples (number of documents, number of words, (number of keywords,...))
		"""
		D = int(np.sum(self.g.vp['kind'].a == 0)) #documents
		W = int(np.sum(self.g.vp['kind'].a == 1)) #words
		K = [int(np.sum(self.g.vp['kind'].a == (k+2))) for k in range(self.nbranches)] #keywords
		return D, W, K

	# Helper functions	  

	def get_groups(self, l=0):

	# --- Numba function for edge processing with list of arrays ---
		@njit
		def process_edges_numba_list(sources, targets, z1, z2, kinds, weights,
				                     D, W, K_arr, nbranches,
				                     n_db, n_wb, n_dbw, n_w_key_b_list, n_dbw_key_list):

			for i in range(len(sources)):
				v1 = sources[i]
				v2 = targets[i]
				w = weights[i]
				t1 = z1[i]
				t2 = z2[i]
				kind = kinds[i]

				n_db[v1, t1] += w

				if kind == 1:
				    n_wb[v2 - D, t2] += w
				    n_dbw[v1, t2] += w
				else:
				    ik = kind - 2
				    offset = D + W
				    for j in range(ik):
				        offset += K_arr[j]
				    n_w_key_b_list[ik][v2 - offset, t2] += w
				    n_dbw_key_list[ik][v1, t2] += w


		if l in self.groups:
			return self.groups[l]

		state_l = self.state.project_level(l).copy(overlap=True)
		state_l_edges = state_l.get_edge_blocks()
		B = state_l.get_B()
		D, W, K = self._get_shape()
		nbranches = self.nbranches

		# Preallocate arrays
		n_wb = np.zeros((W, B))
		n_db = np.zeros((D, B))
		n_dbw = np.zeros((D, B))

		# For branches, use list of arrays (one per branch) to avoid broadcasting issues
		n_w_key_b = [np.zeros((K[ik], B)) for ik in range(nbranches)]
		n_dbw_key = [np.zeros((D, B)) for _ in range(nbranches)]

		# Convert graph edges to arrays
		edges = list(self.g.edges())
		sources = np.array([e.source() for e in edges], dtype=np.int64)
		targets = np.array([e.target() for e in edges], dtype=np.int64)
		weights = np.array([self.g.ep["count"][e] for e in edges], dtype=np.float64)
		z1_arr = np.array([state_l_edges[e][0] for e in edges], dtype=np.int64)
		z2_arr = np.array([state_l_edges[e][1] for e in edges], dtype=np.int64)
		kinds = np.array([self.g.vp['kind'][v] for v in targets], dtype=np.int64)

		# --- Edge processing (Numba-accelerated) ---
		process_edges_numba_list(sources, targets, z1_arr, z2_arr, kinds, weights, D, W, K, nbranches, n_db, n_wb, n_dbw, n_w_key_b, n_dbw_key)

		# --- Keep only nonzero columns safely ---
		ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
		n_db = n_db[:, ind_d]
		Bd = len(ind_d)

		ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
		n_wb = n_wb[:, ind_w]
		Bw = len(ind_w)

		ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
		n_dbw = n_dbw[:, ind_w2]

		Bk = []
		for ik in range(nbranches):
		    ind_wk = np.where(np.sum(n_w_key_b[ik], axis=0) > 0)[0]
		    n_w_key_b[ik] = n_w_key_b[ik][:, ind_wk].copy()
		    Bk.append(len(ind_wk))

		    ind_w2k = np.where(np.sum(n_dbw_key[ik], axis=0) > 0)[0]
		    n_dbw_key[ik] = n_dbw_key[ik][:, ind_w2k].copy()

		# --- Compute probabilities ---
		p_tw_w = (n_wb / np.nansum(n_wb, axis=1)[:, None]).T
		p_tk_w_key = [(n_w_key_b[ik] / np.nansum(n_w_key_b[ik], axis=1)[:, None]).T
		              for ik in range(nbranches)]
		p_w_tw = n_wb / np.nansum(n_wb, axis=0)[None, :]
		p_w_key_tk = [n_w_key_b[ik] / np.nansum(n_w_key_b[ik], axis=0)[None, :]
		              for ik in range(nbranches)]
		p_tw_d = (n_dbw / np.nansum(n_dbw, axis=1)[:, None]).T
		p_tk_d = [(n_dbw_key[ik] / np.nansum(n_dbw_key[ik], axis=1)[:, None]).T
		          for ik in range(nbranches)]
		p_td_d = (n_db / np.nansum(n_db, axis=1)[:, None]).T

		result = {	'Bd': Bd, 'Bw': Bw, 'Bk': Bk,
					'p_tw_w': p_tw_w,
					'p_tk_w_key': p_tk_w_key,
					'p_td_d': p_td_d,
					'p_w_tw': p_w_tw,
					'p_w_key_tk': p_w_key_tk,
					'p_tw_d': p_tw_d,
					'p_tk_d': p_tk_d}

		self.groups[l] = result
		return result

	
	def metadata(self, l=0, n=10, kind=2):
		'''
		get the n most common keywords for each keyword-group in level l.
		
		:return: tuples (keyword,P(kw|tk))
		'''

		dict_groups = self.get_groups(l)
		Bw = dict_groups['Bk'][kind-2]
		p_w_tw = dict_groups['p_w_key_tk'][kind-2]

		words = self.keywords[kind-2]

		## loop over all word-groups
		dict_group_keywords = {}
		for tw in range(Bw):
			p_w_ = p_w_tw[:, tw]
			ind_w_ = np.argsort(p_w_)[::-1]
			list_words_tw = []
			for i in ind_w_[:n]:
				if p_w_[i] > 0:
					list_words_tw += [(words[i], p_w_[i])]
				else:
					break
			dict_group_keywords[tw] = list_words_tw
		return dict_group_keywords

	def metadatumdist(self, doc_index, l=0, kind=2):
		dict_groups = self.get_groups(l)
		p_tk_d = dict_groups['p_tk_d'][kind-2]
		list_topics_tk = []
		for tk, p_tk in enumerate(p_tk_d[:, doc_index]):
			list_topics_tk += [(tk, p_tk)]
		return list_topics_tk


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


	def save_single_level(self, l: int, name: str) -> None:
		"""
		Save per-level probability matrices (topics, clusters, documents) for the given level.

		Parameters
		----------
		l : int
			The level index to save. Must be within the range of available model levels.
		name : str
			Base path (folder + prefix) where files will be written.
			Example: "results/mymodel" → files like:
				- results/mymodel_level_0_mainfeature_topics.tsv.gz
				- results/mymodel_level_0_clusters.tsv.gz
				- results/mymodel_level_0_mainfeature_topics_documents.tsv.gz
				- results/mymodel_level_0_metafeature_topics.tsv.gz
				- results/mymodel_level_0_metafeature_topics_documents.tsv.gz

		Notes
		-----
		- Files are written as tab-separated values (`.tsv.gz`) with gzip compression.
		- Handles both the main feature (`self.modalities[0]`) and any meta-features (`self.modalities[1:]`).
		- Raises RuntimeError if any file cannot be written.
		"""

		# --- Validate inputs ---
		if not isinstance(l, int) or l < 0 or l >= len(self.state.levels):
			raise ValueError(f"Invalid level index {l}. Must be between 0 and {len(self.state.levels) - 1}.")
		if not isinstance(name, str) or not name.strip():
			raise ValueError("`name` must be a non-empty string path prefix.")

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

		# --- P(main_feature | main_topic) ---
		p_w_tw = pd.DataFrame(data=data["p_w_tw"], index=self.words,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		_safe_save(p_w_tw, f"{name}_level_{l}_{main_feature}_topics.tsv.gz")

		# --- P(meta_feature | meta_topic_feature), if any ---
		if len(self.modalities) > 1:
			for k, meta_features in enumerate(self.modalities[1:]):
				feat_topic = pd.DataFrame(data=data["p_w_key_tk"][k], index=self.keywords[k],
					columns=[f"{meta_features}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(feat_topic, f"{name}_level_{l}_{meta_features}_topics.tsv.gz")

			# --- P(document | cluster) ---
			clusters = pd.DataFrame(data=data["p_td_d"], columns=self.documents)
			_safe_save(clusters, f"{name}_level_{l}_clusters.tsv.gz")

			# --- P(main_topic | documents) ---
			p_tw_d = pd.DataFrame(data=data["p_tw_d"].T,index=self.documents,
				columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
			_safe_save(p_tw_d, f"{name}_level_{l}_{main_feature}_topics_documents.tsv.gz")

			# --- P(meta_topic | document) ---
			for k, meta_features in enumerate(self.modalities[1:]):
				p_tk_d = pd.DataFrame(data=data["p_tk_d"][k].T, index=self.documents,
					columns=[f"{meta_features}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				_safe_save(p_tk_d, f"{name}_level_{l}_{meta_features}_topics_documents.tsv.gz")



	def save_data(self, name: str = "results/mymodel") -> None:
		"""
		Save the global graph, model, state, and level-specific data for the current nSBM self.

		Parameters
		----------
		name : str, optional
			Base path (folder + prefix) where all outputs will be saved.
			Example: "results/mymodel" will produce:
				- results/mymodel_graph.xml.gz
				- results/mymodel_self.pkl	
				- results/mymodel_entropy.txt
				- results/mymodel_state.pkl
				- results/mymodel_level_X_*.tsv.gz  (per level, up to 6 levels)

		Notes
		-----
		- The parent folder is created automatically if it does not exist.
		- Level saving is parallelized with threads for efficiency in I/O.
		- By default, at most 6 levels are saved, or fewer if the model has <6 levels.
		- Exceptions in parallel tasks are caught and reported without stopping other tasks.
		"""

		# --- Validate name ---
		if not isinstance(name, str) or not name.strip():
			raise ValueError("`name` must be a non-empty string representing the save path.")

		# --- Ensure folder exists ---
		folder = os.path.dirname(name)
		if folder:
			Path(folder).mkdir(parents=True, exist_ok=True)

		# --- Save global files ---
		try:
			self.save_graph(filename=f"{name}_graph.xml.gz")
			self.dump_model(filename=f"{name}_model.pkl")

			with open(f"{name}_entropy.txt", "w") as f:
				f.write(str(self.state.entropy()))

			with open(f"{name}_state.pkl", "wb") as f:
				pickle.dump(self.state, f)

		except Exception as e:
			raise RuntimeError(f"Failed to save global files for model '{name}': {e}") from e


		# --- Save levels in parallel (threaded to avoid data duplication) ---
		L = min(len(self.state.levels), self.max_depth)
		if L == 0:
			print("Nothing to save")
			return  # nothing to save

		errors = []
		with ThreadPoolExecutor() as executor:
			futures = {executor.submit(self.save_single_level, l, name): l for l in range(L)}
			for future in as_completed(futures):
				l = futures[future]
				try:
					future.result()
				except Exception as e:
					errors.append((l, str(e)))

		if errors:
			msg = "; ".join([f"Level {l}: {err}" for l, err in errors])
			raise RuntimeError(f"Errors occurred while saving levels: {msg}")
