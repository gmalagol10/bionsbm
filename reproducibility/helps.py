import numpy as np
import pandas as pd
import json
import os
import json
import scipy
from time import localtime, strftime
#import episcanpy as epi
import scanpy as sc
import cloudpickle as pickle
import os, time, threading, gc, psutil, sklearn
from sklearn.model_selection import train_test_split
import muon as mu
import scipy.sparse as sp
import anndata as ad

colors_to_use=[(0.34550725069638827, 0.4203708006658883, 0.9696902293486781),
 (0.9893800026041992, 0.378955911742755, 0.21756841368122667),
 (0.3959642074605608, 0.24823947872676938, 0.4016676539297192),
 (0.9937826924994482, 0.4211527500079969, 0.8812994030921271),
 (0.4140058397372807, 0.9619317608252869, 0.3109026417629064),
 (0.2286247431221609, 0.6437632542888629, 0.4081322805120583),
 (0.25003260615661993, 0.938691496932296, 0.9192515923797947),
 (0.7646511697684856, 0.24254983894398235, 0.7085129830496552),
 (0.3017721221187747, 0.6522700618245787, 0.9844707721904342),
 (0.21782892631529166, 0.2854088109905996, 0.7174819557293214),
 (0.9991896546476063, 0.4844986464266022, 0.5344476773522967),
 (0.8919236560192338, 0.7949117963224906, 0.7730486745511909),
 (0.6428145648345002, 0.31108252586505475, 0.2041098261347507),
 (0.20269999368314223, 0.7748472379028853, 0.6824940025160084),
 (0.9819923318374866, 0.6965816490867496, 0.21263663000131983),
 (0.809020033101876, 0.23056728504993104, 0.9541856467035792),
 (0.9191255237720919, 0.20589563833687236, 0.4656154484972612),
 (0.4066816185888487, 0.7781338620666193, 0.22302116197384453),
 (0.9393586762460103, 0.9614694589148117, 0.22619331869433382),
 (0.29226458048518345, 0.42830335364029093, 0.6916335500700534)]
	

colors_to_use_pastel=[(0.5665516830630946, 0.5037138904751852, 0.5075215212500083),
 (0.9949012052279167, 0.7553237006975844, 0.5055095027008268),
 (0.6013710840316495, 0.6764106146805113, 0.9895579386673582),
 (0.998816092724718, 0.7071878014442006, 0.8979090191443285),
 (0.5429870427410571, 0.750719571440085, 0.6124765491924715),
 (0.5084695919264852, 0.5093366344721296, 0.9990436978593071),
 (0.8246194014837736, 0.9846839170837028, 0.5089892119497516),
 (0.5052972476002842, 0.5444186786858283, 0.7812964942323839),
 (0.5124601533942488, 0.9750384643760528, 0.9697621879182152),
 (0.9728598314959997, 0.5013134514042519, 0.8469604310077152)]

def create_count_matrix(fragments_file : str, valid_bcs : list, features_space : str, features_file=None, source=None, metadata=None):

	if not isinstance(fragments_file, str) or not isinstance(valid_bcs, list) or not isinstance(features_space, str):
		print(type(fragments_file), type(valid_bcs), type(features_space), flush=True)
		raise TypeError
	
	if features_space == "Window":
		adata = epi.ct.window_mtx(fragments_file=fragments_file, valid_bcs=valid_bcs)
		
	elif features_space != "Window":
		if features_file[-6:]=="gtf.gz" or features_file[-3:]=="gtf":
			if features_space=="GA":
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs)
			else:
				data={}
				data["tRNA"]=[30,80] #POL3
				data["rRNA"]=[30,80] #POL1/POL3
				data["protein_coding"]=[5000,2000] #POL2
				data["lncRNA"]=[30,80] #POL3
				data["miRNA"]=[5000,2000] #POl2
				data["telomerase_RNA"]=[5000,2000] #POL2
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs, 
								 upstream=data[features_space][0], downstream=data[features_space][1], 
								 source=source, gene_type=[features_space])
								 	
		elif features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed":
			adata = epi.ct.peak_mtx(fragments_file=fragments_file, peak_file=features_file, valid_bcs=valid_bcs, normalized_peak_size=None)
			adata.var.index=["_".join([str(adata.var.iloc[i][0]),str(adata.var.iloc[i][1]),str(adata.var.iloc[i][2])]) for i in range(len(adata.var.index))]
	
	adata.X=scipy.sparse.csr_matrix(adata.X, dtype="float32")
	adata.var.columns = adata.var.columns.astype(str)
	adata.obs.columns = adata.obs.columns.astype(str)
	adata.var = adata.var.rename(columns={"0": "chr", "1": "start", "2" : "stop"})
	adata.var_names_make_unique(join="_")
	adata = adata[:,adata.var[adata.var['chr'].str.match('chr')].index]
	adata = adata[:,adata.var[~adata.var['chr'].str.match('chrM')].index]
	if features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed" or features_space == "Window":
		a = [a for a in adata.var.index if len(a.split("_"))!=3]
		adata = adata[ :, adata.var.loc[list(set(adata.var.index) - set(a))].index ]
		del a
	adata.obs = metadata.loc[adata.obs.index]

	return adata
  
def qc_filtering(adata, omic="ATAC"):
	max_nucleosome_signal = 4
	min_tss_enrichment = 2

	epi.pp.qc_stats(adata, verbose=False)
		
	if omic=="GEX":
		adata.var["MT"] = adata.var.index.str.startswith(("MT","mt"))
		sc.pp.calculate_qc_metrics(adata, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
		
	min_features = 10**np.quantile(adata.obs["log_n_features"], 0.05)
	max_features = 10**np.quantile(adata.obs["log_n_features"], 0.95)
	min_cells = 10**np.quantile(adata.var["log_n_cells"], 0.05)
	max_cells = 10**np.quantile(adata.var["log_n_cells"], 0.95)

	print("Adata's shape:", adata.shape, flush=True)
	
	epi.pp.set_filter(adata, "n_features", min_threshold=min_features, max_threshold=max_features, verbose=True)
	epi.pp.set_filter(adata, "n_cells", min_threshold=min_cells, max_threshold=max_cells, verbose=True)
	if omic == "ATAC":
		epi.pp.set_filter(adata, "nucleosome_signal", max_threshold=max_nucleosome_signal, verbose=True)
		epi.pp.set_filter(adata, "tss_enrichment_score", min_threshold=min_tss_enrichment, verbose=True)	
	adata = epi.pp.apply_filters(adata, verbose=False)
	print("Adata's shape after cells and features filtering:", adata.shape, flush=False)
	
	if omic=="GEX":
		max_mt = np.quantile(adata.obs["pct_counts_MT"].dropna(), 0.9)
		adata = adata[adata.obs["pct_counts_MT"]<=max_mt]
		print("Adata's shape after MT filtering:", adata.shape, flush=True)
		
		adata=adata[adata.obs.predicted_doublet==False]
		print("Adata's shape after doublets filtering:", adata.shape, flush=True)
   
	epi.pp.normalize_total(adata)
		
	sc.pp.highly_variable_genes(adata, flavor='seurat_v3')
	if omic=="GEX":
		min_var = np.quantile(adata.var.variances_norm, 0.9)
	else:
		min_var = np.quantile(adata.var.variances_norm, 0.7)
	adata = adata[:, adata.var.variances_norm>min_var]
	print("Adata's shape after HVG filtering:", adata.shape, flush=True)
	
	epi.pp.log1p(adata)

	adata = adata[:, adata.X.max(axis=0)>0]
	adata = adata[adata.X.max(axis=1)>0]
	print("Adata's shape after max 0 filtering:", adata.shape, flush=True)
	   
	return adata 
 
		
def preprocessing(adata, omic="ATAC"):
	print("QC and filtering", flush=True)

	adata=qc_filtering(adata, omic=omic)
	
	print("PCA, UMAP and clustering", flush=True)
	sc.pp.pca(adata)
	sc.pp.neighbors(adata, method="umap")
	sc.tl.umap(adata)
	sc.tl.leiden(adata)

	return adata
  
   
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))


def flat_list(l):
	return np.array(list(set([item for sublist in l for item in sublist])))


def most_common(l):
	l=list(l)
	return max(set(l), key=l.count)


def create_alg(word):
	return word.split("_")[0]
		
def create_data(word):
	return "_".join(word.split("_")[1:])

def find_best_level(results, ground_truth, label="CellType", method="best", metric="NMI/NMI*"):
	ok_levels, exps = {}, {}
	for exp in sorted(set(results["Exp"])):
		ok_levels[exp] = {}
		if method=="CT_clusters":
			for run in range(0,25):
				exps[exp] = {}
				for level in [0,1,2,3]:
					a=results[(results["Level"]==int(level)) & (results["GT"]==label) & (results["Exp"]==exp) & (results["Run"]==run)][["N_clusters"]]
					if len(a["N_clusters"]) > 0:
						exps[exp][str(level)]=float(np.abs(a["N_clusters"]-len(set(ground_truth[label].dropna()))))
				if len(exps[exp]) > 0:
					ok_levels[exp][run]=np.argmin(pd.DataFrame.from_dict(exps[exp], orient="index").fillna(10000000000))
		elif method=="best":
			for run in range(0,25):
				exps[exp] = {}
				for level in [0,1,2,3]:
					a=results[(results["Level"]==int(level)) & (results["GT"]==label) & (results["Exp"]==exp) & (results["Run"]==run)][[metric]]
					if len(a[metric]) > 0:
						exps[exp][str(level)]=float(a[metric])
				if len(exps[exp]) > 0:
					ok_levels[exp][run]=np.argmax(pd.DataFrame.from_dict(exps[exp], orient="index").fillna(0))			
	return ok_levels

def create_results(dataset, label, metric="NMI/NMI*"):
	
	SBMs=pd.read_csv(f"Datasets/{dataset}/SBM/{dataset}_SBMs_25Run_NMI.tsv.gz", sep="\t", index_col=0)
	SBMs=SBMs[SBMs["GT"]==label]
	SBMs["Algorithm"]=[create_alg(SBMs.iloc[i]["Exp"]) for i in range(len(SBMs))]
	SBMs["Data"]=[create_data(SBMs.iloc[i]["Exp"]) for i in range(len(SBMs))]
	gt=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
	ok_levels_best=find_best_level(SBMs, gt, label=label, method="best", metric=metric)
	ok_levels_nc=find_best_level(SBMs, gt, label=label, method="CT_clusters", metric=metric)


	SBMs_def_best=pd.DataFrame()
	for exp in ok_levels_best.keys():
		for run in ok_levels_best[exp].keys():
			d=SBMs[(SBMs["Exp"]==exp) & (SBMs["Run"]==run) & (SBMs["Level"]==int(ok_levels_best[exp][run]))]
			SBMs_def_best=pd.concat([SBMs_def_best, d])

	SBMs_def_ct=pd.DataFrame()
	for exp in ok_levels_nc.keys():
		for run in ok_levels_nc[exp].keys():
			d=SBMs[(SBMs["Exp"]==exp) & (SBMs["Run"]==run) & (SBMs["Level"]==int(ok_levels_nc[exp][run]))]
			SBMs_def_ct=pd.concat([SBMs_def_ct, d])
	
   
	mowgli=pd.read_csv(f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_25Run_NMI.tsv.gz", sep="\t", index_col=0)
	mowgli=mowgli[mowgli["GT"]==label]
	mowgli["Algorithm"]="Mowgli"
	mowgli["Data"]=["_".join(mowgli.iloc[i]["Exp"].split("_")[1:]) for i in range(len(mowgli))]	

	if dataset not in ["BMMCCite", "Spleen"]:
		ShareTopic=pd.read_csv(f"../Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_25Run_NMI.tsv.gz", sep="\t", index_col=0)
		ShareTopic=ShareTopic[ShareTopic["GT"]==label]
		ShareTopic["Algorithm"]="ShareTopic"
		ShareTopic["Data"]=["_".join(ShareTopic.iloc[i]["Exp"].split("_")[1:]) for i in range(len(ShareTopic))]
	else:
		ShareTopic=pd.DataFrame()
	
	df_best=pd.concat([SBMs_def_best, ShareTopic, mowgli], axis=0)
	df_best.reset_index(inplace=True)
	df_best.drop("index", axis=1, inplace=True)
	df_best["Data"]=df_best["Data"].replace("_", "+")

	df_ct=pd.concat([SBMs_def_ct, ShareTopic, mowgli], axis=0)
	df_ct.reset_index(inplace=True)
	df_ct.drop("index", axis=1, inplace=True)
	df_ct["Data"]=df_ct["Data"].replace("_", "+")

	return df_best, df_ct


def create_results_topics(dataset, exps=None):
	
	SBMs=pd.read_csv(f"Datasets/{dataset}/SBM/{dataset}_SBMs_25Run_TopicsSpec.tsv.gz", sep="\t", index_col=0)
	SBMs["Algorithm"]=[create_alg(SBMs.iloc[i]["Exp"]) for i in range(len(SBMs))]
	SBMs["Data"]=[create_data(SBMs.iloc[i]["Exp"]) for i in range(len(SBMs))]   
  
	mowgli=pd.read_csv(f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_25Run_TopicsSpec.tsv.gz", sep="\t", index_col=0)
	mowgli["Algorithm"]="Mowgli"
	mowgli["Data"]=["_".join(mowgli.iloc[i]["Exp"].split("_")[1:]) for i in range(len(mowgli))]	
	mowgli["FS"]="Mixed"
		
	if dataset not in ["BMMCCite", "Spleen"]:
		ShareTopic=pd.read_csv(f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_25Run_TopicsSpec.tsv.gz", sep="\t", index_col=0)
		ShareTopic["FS"]="Mixed"
		ShareTopic["Algorithm"]="ShareTopic"
		ShareTopic["Data"]=["_".join(ShareTopic.iloc[i]["Exp"].split("_")[1:]) for i in range(len(ShareTopic))]
		if exps is not None:
			ShareTopic=ShareTopic[ShareTopic["Exp"].isin(exps)]
			mowgli=mowgli[mowgli["Exp"].isin(exps)]
			SBMs=SBMs[SBMs["Exp"].isin(exps)]
	else:
		ShareTopic=pd.DataFrame()
		if exps is not None:
			mowgli=mowgli[mowgli["Exp"].isin(exps)]
			SBMs=SBMs[SBMs["Exp"].isin(exps)]
	
	df=pd.concat([SBMs, ShareTopic, mowgli], axis=0)
	df.reset_index(inplace=True)
	df.drop("index", axis=1, inplace=True)
	df["Data"]=df["Data"].replace("_", "+")
	
	return df



# ---------- Helper: peak RSS (incl. child processes) while a callable runs ----------
def _rss_now(proc: psutil.Process, include_children: bool = True) -> int:
	"""Return current RSS in bytes for proc (+ children if include_children)."""
	rss = 0
	try:
		rss = proc.memory_info().rss
	except psutil.Error:
		pass
	if include_children:
		for c in proc.children(recursive=True):
			try:
				rss += c.memory_info().rss
			except psutil.Error:
				pass
	return rss

def run_with_peak_increase(fn, *args, poll_s: float = 0.01, include_children: bool = True, discard_result: bool = False, **kwargs):
	"""
	Measure ONLY the function's incremental RAM footprint.
	Returns: (result_or_None, time_sec, peak_increase_bytes)

	- Baseline is sampled right before starting the thread.
	- We poll RSS and report max(RSS - baseline).
	- If discard_result=True, the function's return value is immediately dropped
	  to avoid counting persistently stored outputs as "algorithmic" usage.
	"""
	gc.collect()  # reduce noise from previous work
	proc = psutil.Process(os.getpid())

	def rss_now():
		rss = 0
		try: rss = proc.memory_info().rss
		except psutil.Error: pass
		if include_children:
			for c in proc.children(recursive=True):
				try: rss += c.memory_info().rss
				except psutil.Error: pass
		return rss

	# Baseline: after GC, before the function starts
	baseline = rss_now()
	peak_increase = 0
	done, out, err = False, None, None

	def runner():
		nonlocal out, err, done
		try:
			out = fn(*args, **kwargs)
		except BaseException as e:
			err = e
		finally:
			done = True

	t = threading.Thread(target=runner)
	t.start()
	start = time.time()
	while not done:
		inc = rss_now() - baseline
		if inc > peak_increase:
			peak_increase = inc
		time.sleep(poll_s)
	t.join()
	elapsed = time.time() - start

	# Final sample in case the last spike is right before return
	inc = rss_now() - baseline
	if inc > peak_increase:
		peak_increase = inc

	if discard_result:
		# Drop the function's return value so persistent outputs aren't counted
		out = None
		gc.collect()

	if err:
		raise err
	return out, elapsed, max(0, peak_increase)



def muon_subsample(mdata, subsample_dict, strata_column, random_state=42):
	"""
	Build a new MuData object from scratch to avoid indexing issues
	Preserves .layers["raw"] if it exists
	"""
	np.random.seed(random_state)
	
	modality_names = list(mdata.mod.keys())
	
	print(f"Original data: {mdata.n_obs} cells")
	
	# Step 1: Cell subsampling FIRST
	if 'cells' in subsample_dict and subsample_dict['cells'] is not None:
		n_cells = subsample_dict['cells']
		
		print(f"Step 1: Subsampling to {n_cells} cells...")
		
		if strata_column in mdata.obs.columns:
			strata_labels = mdata.obs[strata_column]
		else:
			strata_labels = mdata[modality_names[0]].obs[strata_column]
		
		try:
			_, cell_indices = train_test_split(
				np.arange(mdata.n_obs),
				test_size=n_cells,
				stratify=strata_labels,
				random_state=random_state
			)
			# Create a new MuData with subsampled cells
			mod_dict = {}
			for mod in modality_names:
				mod_data = mdata[mod][cell_indices].copy()
				mod_dict[mod] = mod_data
			mdata_subsampled = mu.MuData(mod_dict)
		except ValueError as e:
			print(f"Stratified sampling failed: {e}")
			print("Falling back to random sampling...")
			cell_indices = np.random.choice(mdata.n_obs, n_cells, replace=False)
			mod_dict = {}
			for mod in modality_names:
				mod_data = mdata[mod][cell_indices].copy()
				mod_dict[mod] = mod_data
			mdata_subsampled = mu.MuData(mod_dict)
	else:
		print("Step 1: No cell subsampling requested")
		mdata_subsampled = mdata.copy()
	
	print(f"After cell subsampling: {mdata_subsampled.n_obs} cells")
	
	# Step 2: Feature subsampling
	print("Step 2: Feature subsampling...")
	for modality in modality_names:
		if modality in subsample_dict and subsample_dict[modality] is not None:
			n_features = subsample_dict[modality]
			mod_data = mdata_subsampled[modality]
			
			if n_features < mod_data.n_vars:
				feature_indices = np.random.choice(mod_data.n_vars, n_features, replace=False)
				
				# Create a completely new AnnData object preserving layers
				new_X = mod_data.X[:, feature_indices]
				new_var = mod_data.var.iloc[feature_indices].copy()
				
				# Copy all layers, subsampling the "raw" layer if it exists
				new_layers = {}
				for layer_name in mod_data.layers.keys():
					if layer_name == "raw":
						new_layers["raw"] = mod_data.layers["raw"][:, feature_indices]
					else:
						new_layers[layer_name] = mod_data.layers[layer_name][:, feature_indices]
				
				new_adata = ad.AnnData(
					X=new_X,
					obs=mod_data.obs.copy(),
					var=new_var,
					obsm=mod_data.obsm.copy(),
					varm=mod_data.varm.copy() if hasattr(mod_data, 'varm') else {},
					uns=mod_data.uns.copy(),
					layers=new_layers
				)
				mdata_subsampled.mod[modality] = new_adata
				print(f"  {modality}: {n_features} features (preserved raw layer)")
			else:
				print(f"  {modality}: {mod_data.n_vars} features (no subsampling)")
		else:
			n_current = mdata_subsampled[modality].n_vars
			print(f"  {modality}: {n_current} features (no specification)")
#	for mod in mdata_subsampled.mod.keys():
#		total_counts = np.array(mdata_subsampled[mod].X.sum(axis=1)).flatten()
		
		# Find cells with zero counts
#		zero_count_cells = total_counts == 0
#		zero_count_indices = np.where(zero_count_cells)[0]
#		
#		print(f"{mod} --> Total cells: {mdata_subsampled[mod].n_obs}")
#		print(f"{mod} --> Cells with zero counts: {np.sum(zero_count_cells)}")
#		print(f"{mod} -->  Indices of zero-count cells: {zero_count_indices}")

	return mdata_subsampled

