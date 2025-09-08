"""
bionsbm

Copyright(C) 2021 fvalle1

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

import graph_tool.all as gt
import numpy as np
import pandas as pd
import cloudpickle as pickle
import os, sys
import muon as mu
import scanpy as sc
import functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


"""
Inherit hSBM code from https://github.com/martingerlach/hSBM_Topicmodel
"""
from . import sbmtm


class bionsbm(sbmtm.sbmtm):
	"""
	Class to run bionsbm
	"""
	def __init__(self, obj, label=None):
		super().__init__()
		self.keywords = []
		self.nbranches = 1
		self.modalities = []

		if isinstance(obj, mu.MuData):
			self.modalities=list(obj.mod.keys())   
			dfs=[obj[key].to_df().T for key in self.modalities]
			self.make_graph_multiple_df(dfs[0], dfs[1:])

		elif isinstance(obj, ad.AnnData):
			self.modalities=["Mod1"]
			self.make_graph_multiple_df(dfs[0])

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
		Load a presaved graph

		:param filename: graph to load
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
	   
		return self.make_graph(df_all.drop("kind", axis=1), get_kind)
		
	def make_graph(self, df: pd.DataFrame, get_kind)->None:
		"""
		Create a graph from a pandas DataFrame

		:param df: DataFrame with words on index and texts on columns. Actually this is a BoW.
		:param get_kind: function that returns 1 or 2 given an element of df.index. [1 for words 2 for keywords]
		"""
		self.g = gt.Graph(directed=False)
		name = self.g.vp["name"] = self.g.new_vp("string")
		kind = self.g.vp["kind"] = self.g.new_vp("int")
		weight = self.g.ep["count"] = self.g.new_ep("int")
		
		for doc in df.columns:
			d = self.g.add_vertex()
			name[d] = doc
			kind[d] = 0
			
		for word in df.index:
			w = self.g.add_vertex()
			name[w] = word
			kind[w] = get_kind(word)

		D = df.shape[1]
		
		for i_doc, doc in enumerate(df.columns):
			text = df[doc]
			self.g.add_edge_list([(i_doc,D + x[0][0],x[1]) for x in zip(enumerate(df.index),text)], eprops=[weight])

		filter_edges = self.g.new_edge_property("bool")
		for e in self.g.edges():
			filter_edges[e] = weight[e]>0

		self.g.set_edge_filter(filter_edges)
		self.g.purge_edges()
		self.g.clear_filters()
		
		self.documents = df.columns
		self.words = df.index[self.g.vp['kind'].a[D:] == 1]
		for ik in range(2,2+self.nbranches):# 2 is doc and words
			self.keywords.append(df.index[self.g.vp['kind'].a[D:] == ik])
		
	def fit(self, n_init=1, verbose=True, deg_corr=True, overlap=False, parallel=False, B_min=3, B_max=None, clabel=None, *args, **kwargs) -> None:
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
		print(state_args, flush=True)

		for _ in range(n_init):
			print("Fit number:", _, flush=True)
			state = gt.minimize_nested_blockmodel_dl(self.g,
													state_args=state_args,
													multilevel_mcmc_args={
														"B_min": B_min,
														"B_max": B_max,
														"verbose": verbose,
														"parallel" : parallel
													},
									*args, 
									**kwargs)
			
			entropy = state.entropy()
			if entropy < min_entropy:
				min_entropy = entropy
				self.state = state
				
		self.mdl = min_entropy

		L = len(self.state.levels)
		dict_groups_L = {}
		self.groups = dict_groups_L
		"""	  
		## only trivial bipartite structure
		if L == 2:
			self.L = 1
			for l in range(L - 1):
				dict_groups_l = self.get_groups(l=l)
				dict_groups_L[l] = dict_groups_l
		## omit trivial levels: l=L-1 (single group), l=L-2 (tripartite)
		else:
			self.L = L - 2
			for l in range(L - 2):
				dict_groups_l = self.get_groups(l=l)
				dict_groups_L[l] = dict_groups_l
		self.groups = dict_groups_L
		"""



	def dump_model(self, filename="bionsbm.pkl"):
		"""
		Dump model using pickle

		To restore the model:

		import cloudpickle as pickle
		file=open(\"bionsbm.pkl\" ,\"rb\")
		model = pickle.load(file)

		file.close()
		"""
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

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
		"""
		return groups

		:param l: hierarchy level
		"""

		#sort of cache if groups are already estimated avoid re running
		if l in self.groups.keys():
			return self.groups[l]

		state_l = self.state.project_level(l).copy(overlap=True)
		state_l_edges = state_l.get_edge_blocks()
		B = state_l.get_B()
		D, W, K = self._get_shape()

		n_wb = np.zeros((W, B))  ## number of half-edges incident on word-node w and labeled as word-group tw
		n_w_key_b = [np.zeros((K[ik], B)) for ik in range(self.nbranches)]  ## number of half-edges incident on word-node w and labeled as word-group tw
		n_db = np.zeros((D, B))  ## number of half-edges incident on document-node d and labeled as document-group td
		n_dbw = np.zeros((D, B)) ## number of half-edges incident on document-node d and labeled as word-group tw
		n_dbw_key = [np.zeros((D, B)) for _ in range(self.nbranches)] ## number of half-edges incident on document-node d and labeled as keyword-group tw_key

		for e in self.g.edges():
			z1, z2 = state_l_edges[e]
			v1 = e.source()
			v2 = e.target()
			weight = self.g.ep["count"][e]
			n_db[int(v1), z1] += weight
			kind = self.g.vp['kind'][v2]
			if kind == 1:
				n_wb[int(v2) - D, z2] += weight
				n_dbw[int(v1), z2] += weight
			else:
				n_w_key_b[kind-2][int(v2) - D - W - sum(K[:(kind-2)]), z2] += weight
				n_dbw_key[kind-2][int(v1), z2] += weight

		#p_w = np.sum(n_wb, axis=1) / float(np.sum(n_wb))

		ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
		Bd = len(ind_d)
		n_db = n_db[:, ind_d]

		ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
		Bw = len(ind_w)
		n_wb = n_wb[:, ind_w]

		ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
		n_dbw = n_dbw[:, ind_w2]

		ind_w_key = []
		ind_w2_keyword = []
		Bk = []

		for ik in range(self.nbranches):
			ind_w_key.append(np.where(np.sum(n_w_key_b[ik], axis=0) > 0)[0])
			Bk.append(len(ind_w_key[ik]))
			n_w_key_b[ik] = n_w_key_b[ik][:, ind_w_key[ik]]
			
			ind_w2_keyword.append(np.where(np.sum(n_dbw_key[ik], axis=0) > 0)[0])
			n_dbw_key[ik] = n_dbw_key[ik][:, ind_w2_keyword[ik]]
		

		# group membership of each word-node P(t_w | w)
		p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

		p_tk_w_key = []
		for ik in range(self.nbranches):
			# group membership of each keyword-node P(t_k | keyword)
			p_tk_w_key.append((n_w_key_b[ik] / np.sum(n_w_key_b[ik], axis=1)[:, np.newaxis]).T)
		
		## topic-distribution for words P(w | t_w)
		p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]
		
		p_w_key_tk = []
		for ik in range(self.nbranches):
			## topickey-distribution for keywords P(keyword | t_w_key)
			p_w_key_tk.append(n_w_key_b[ik] / np.sum(n_w_key_b[ik], axis=0)[np.newaxis, :])
		
		## Mixture of word-groups into documetns P(t_w | d)
		p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

		p_tk_d = []
		for ik in range(self.nbranches):
			## Mixture of word-groups into documetns P(t_w | d)
			p_tk_d.append((n_dbw_key[ik] / np.sum(n_dbw_key[ik], axis=1)[:, np.newaxis]).T)
		
		# group membership of each doc-node P(t_d | d)
		p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

		result = {}
		result['Bd'] = Bd
		result['Bw'] = Bw
		result['Bk'] = Bk
		result['p_tw_w'] = p_tw_w
		result["p_tk_w_key"] = p_tk_w_key
		result['p_td_d'] = p_td_d
		result['p_w_tw'] = p_w_tw
		result['p_w_key_tk'] = p_w_key_tk
		result['p_tw_d'] = p_tw_d
		result['p_tk_d'] = p_tk_d

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
	
	def print_topics(self, l=0, format='csv', path_save=''):
		'''
		Print topics, topic-distributions, and document clusters for a given level in the hierarchy.
		
		:param l: level to store
		:param format: csv (default) or html
		:param path_save: path/to/store/file
		'''
		D, W, K = self._get_shape()

		## topics
		dict_topics = self.topics(l=l, n=-1)

		list_topics = sorted(list(dict_topics.keys()))
		list_columns = ['Topic %s' % (t + 1) for t in list_topics]

		T = len(list_topics)
		df = pd.DataFrame(columns=list_columns, index=range(W))

		for t in list_topics:
			list_w = [h[0] for h in dict_topics[t]]
			V_t = len(list_w)
			df.iloc[:V_t, t] = list_w
		df = df.dropna(how='all', axis=0)
		if format == 'csv':
			fname_save = 'bionsbm_level_%s_topics.csv' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_csv(filename, index=False, na_rep='')
		elif format == 'html':
			fname_save = 'bionsbm_level_%s_topics.html' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_html(filename, index=False, na_rep='')
		elif format == 'tsv':
			fname_save = 'bionsbm_level_%s_topics.tsv' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_csv(filename, index=False, na_rep='', sep='\t')
		else:
			pass

		## topic distributions
		list_columns = ['i_doc', 'doc'] + ['Topic %s' % (t + 1) for t in list_topics]
		df = pd.DataFrame(columns=list_columns, index=range(D))
		for i_doc in range(D):
			list_topicdist = self.topicdist(i_doc, l=l)
			df.iloc[i_doc, 0] = i_doc
			df.iloc[i_doc, 1] = self.documents[i_doc]
			df.iloc[i_doc, 2:] = [h[1] for h in list_topicdist]
		df = df.dropna(how='all', axis=1)
		if format == 'csv':
			fname_save = 'bionsbm_level_%s_topic-dist.csv' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_csv(filename, index=False, na_rep='')
		elif format == 'html':
			fname_save = 'bionsbm_level_%s_topic-dist.html' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_html(filename, index=False, na_rep='')
		else:
			pass
		
		## keywords
		for ik in range(2,2+self.nbranches):
			dict_metadata = self.metadata(l=l, n=-1, kind=ik)

			list_metadata = sorted(list(dict_metadata.keys()))
			list_columns = ['Metadatum %s' % (t + 1) for t in list_metadata]

			T = len(list_topics)
			df = pd.DataFrame(columns=list_columns, index=range(K[ik-2]))

			for t in list_metadata:
				list_w = [h[0] for h in dict_metadata[t]]
				V_t = len(list_w)
				df.iloc[:V_t, t] = list_w
			df = df.dropna(how='all', axis=0)
			if format == 'csv':
				fname_save = 'bionsbm_level_%s_kind_%s_metadata.csv' % (l,ik)
				filename = os.path.join(path_save, fname_save)
				df.to_csv(filename, index=False, na_rep='')
			elif format == 'html':
				fname_save = 'bionsbm_level_%s_kind_%s_metadata.html' % (l,ik)
				filename = os.path.join(path_save, fname_save)
				df.to_html(filename, index=False, na_rep='')
			elif format == 'tsv':
				fname_save = 'bionsbm_level_%s_kind_%s_metadata.tsv' % (l,ik)
				filename = os.path.join(path_save, fname_save)
				df.to_csv(filename, index=False, na_rep='', sep='\t')
			else:
				pass

			## metadata distributions
			list_columns = ['i_doc', 'doc'] + ['Metadatum %s' % (t + 1) for t in list_metadata]
			df = pd.DataFrame(columns=list_columns, index=range(D))
			for i_doc in range(D):
				list_topicdist = self.metadatumdist(i_doc, l=l, kind=ik)
				df.iloc[i_doc, 0] = i_doc
				df.iloc[i_doc, 1] = self.documents[i_doc]
				df.iloc[i_doc, 2:] = [h[1] for h in list_topicdist]
			df = df.dropna(how='all', axis=1)
			if format == 'csv':
				fname_save = 'bionsbm_level_%s_kind_%s_metadatum-dist.csv' % (l,ik)
				filename = os.path.join(path_save, fname_save)
				df.to_csv(filename, index=False, na_rep='')
			elif format == 'html':
				fname_save = 'bionsbm_level_%s_kind_%s_metadatum-dist.html' % (
					l,ik)
				filename = os.path.join(path_save, fname_save)
				df.to_html(filename, index=False, na_rep='')
			else:
				pass

		## doc-groups

		dict_clusters = self.clusters(l=l, n=-1)

		list_clusters = sorted(list(dict_clusters.keys()))
		list_columns = ['Cluster %s' % (t + 1) for t in list_clusters]

		T = len(list_clusters)
		df = pd.DataFrame(columns=list_columns, index=range(D))

		for t in list_clusters:
			list_d = [h[0] for h in dict_clusters[t]]
			D_t = len(list_d)
			df.iloc[:D_t, t] = list_d
		df = df.dropna(how='all', axis=0)
		if format == 'csv':
			fname_save = 'bionsbm_level_%s_clusters.csv' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_csv(filename, index=False, na_rep='')
		elif format == 'html':
			fname_save = 'bionsbm_level_%s_clusters.html' % (l)
			filename = os.path.join(path_save, fname_save)
			df.to_html(filename, index=False, na_rep='')
		else:
			pass

		## word-distr
		list_topics = np.arange(len(self.get_groups(l)['p_w_tw'].T))
		list_columns = ["Topic %d" % (t + 1) for t in list_topics]

		pwtw_df = pd.DataFrame(data=self.get_groups(l)['p_w_tw'], index=self.words, columns=list_columns)
		pwtw_df.replace(0, np.nan)
		pwtw_df = pwtw_df.dropna(how='all', axis=0)
		pwtw_df.replace(np.nan, 0)
		if format == 'csv':
			fname_save = "bionsbm_level_%d_word-dist.csv" % l
			filename = os.path.join(path_save, fname_save)
			pwtw_df.to_csv(filename, index=True, header=True, na_rep='')
		elif format == 'html':
			fname_save = "bionsbm_level_%d_word-dist.html" % l
			filename = os.path.join(path_save, fname_save)
			pwtw_df.to_html(filename, index=True, na_rep='')
		else:
			pass
		
		
		## keyword-distr
		for ik in range(2, 2+self.nbranches):
			list_topics = np.arange(len(self.get_groups(l)['p_w_key_tk'][ik-2].T))
			list_columns = ["Metadatum %d" % (t + 1) for t in list_topics]

			pw_key_tk_df = pd.DataFrame(data=self.get_groups(l)['p_w_key_tk'][ik-2], index=self.keywords[ik-2], columns=list_columns)
			pw_key_tk_df.replace(0, np.nan)
			pw_key_tk_df = pw_key_tk_df.dropna(how='all', axis=0)
			pw_key_tk_df.replace(np.nan, 0)
			if format == 'csv':
				fname_save = "bionsbm_level_%d_kind_%s_keyword-dist.csv" % (l,ik)
				filename = os.path.join(path_save, fname_save)
				pw_key_tk_df.to_csv(filename, index=True, header=True, na_rep='')
			elif format == 'html':
				fname_save = "bionsbm_level_%d_kind_%s_keyword-dist.html" % (l,ik)
				filename = os.path.join(path_save, fname_save)
				pw_key_tk_df.to_html(filename, index=True, na_rep='')
			else:
				pass

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

	def save_single_level(self, l, name):

		main_feature=self.modalities[0]
		data=self.get_groups(l)
		# P(main_feature | main_topic)
		p_w_tw = pd.DataFrame(
			data=data["p_w_tw"],
			index=self.words,
			columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
		p_w_tw.to_csv(f"{name}_level_{l}_{main_feature}_topics.tsv.gz", compression="gzip", sep="\t")

		# P(meta_feature | meta_topic_feature) for each type of meta_feature
		if len(self.modalities) > 1:
			for k, meta_features in enumerate(self.modalities[1:]):
				feat_topic = pd.DataFrame(data=data["p_w_key_tk"][k], index=self.keywords[k],
					columns=[f"{meta_features}_topic_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				feat_topic.to_csv(f"{name}_level_{l}_{meta_features}_topics.tsv.gz", compression="gzip", sep="\t")

			# P(document | cluster)
			pd.DataFrame(data=data["p_td_d"], columns=self.documents).to_csv(f"{name}_level_{l}_clusters.tsv.gz", compression="gzip", sep="\t")

			# P(main_topic | documents)
			p_tw_d = pd.DataFrame(data=data["p_tw_d"].T, index=self.documents,
				columns=[f"{main_feature}_topic_{i}" for i in range(data["p_w_tw"].shape[1])])
			p_tw_d.to_csv(f"{name}_level_{l}_{main_feature}_topics_documents.tsv.gz", compression="gzip", sep="\t")

			# P(meta_topic | document) for each kind of meta_feature
			for k, meta_features in enumerate(self.modalities[1:]):
				p_tk_d = pd.DataFrame(data=data["p_tk_d"][k].T, index=self.documents,
					columns=[f"{meta_features}_topics_{i}" for i in range(data["p_w_key_tk"][k].shape[1])])
				p_tk_d.to_csv(f"{name}_level_{l}_{meta_features}_topics_documents.tsv.gz", compression="gzip", sep="\t")


	def save_data_new(self, name="MyBionSBM/mymodel"):

		#Save global files
		folder="/".join(name.split("/")[:-1])
		Path(folder).mkdir(parents=True, exist_ok=True)

		self.save_graph(filename=f"{name}_graph.xml.gz")
		self.dump_model(filename=f"{name}_model.pkl")
		with open(f"{name}_entropy.txt", "w") as f:
			f.write(str(self.state.entropy()))
		with open(f"{name}_state.pkl", "wb") as f:
			pickle.dump(self.state, f)

		# Parallelise levels
		L = np.min([len(self.state.levels), 6])
		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(self.save_single_level, l, name) for l in range(L)]
			for f in futures:
				f.result()  # wait for all to finish

