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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nmi

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

def _read_metrics(path):
	"""Read a metrics table saved either with or without a pandas index."""
	if not os.path.isfile(path):
		print(f"Metrics file not found: {path}", flush=True)
		return pd.DataFrame()
	df = pd.read_csv(path, sep="\t")
	return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]


def _load_method_results(dataset, folder, label, algorithm, clustering=None):
	path = f"Datasets/{dataset}/{folder}/{dataset}_NMI.tsv.gz"
	df = _read_metrics(path)
	if df.empty:
		return df

	missing = {"Exp", "GT", "NMI", "NMI*"} - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
	if clustering is not None:
		if "Clustering" not in df.columns:
			raise ValueError(f"Missing Clustering column in {path}")
		df = df[df["Clustering"].astype(str).str.lower() == clustering.lower()]

	df = df[df["GT"] == label].copy()
	df["NMI/NMI*"] = df["NMI"] / df["NMI*"].replace(0, np.nan)
	df["Algorithm"] = algorithm
	df["Data"] = df["Exp"].astype(str).str.split("_", n=1).str[-1]
	return df


def create_results(dataset, label, metric="NMI/NMI*", embedding_clustering="KMeans"):
	# bionSBM has multiple hierarchy levels and therefore needs level selection.
	path = f"Datasets/{dataset}/bionSBM/{dataset}_NMI.tsv.gz"
	SBMs = _read_metrics(path)
	SBMs["NMI/NMI*"] = SBMs["NMI"] / SBMs["NMI*"].replace(0, np.nan)
	SBMs = SBMs[SBMs["GT"] == label].copy()
	SBMs["Algorithm"] = "bionSBM"
	SBMs["Data"] = SBMs["Exp"].astype(str).str.split("_", n=1).str[-1]

	gt = pd.read_csv(
		f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0
	)
	best = find_best_level(SBMs, gt, label=label, method="best", metric=metric)
	closest = find_best_level(SBMs, gt, label=label, method="CT_clusters", metric=metric)

	def select_levels(levels):
		rows = [
			SBMs[(SBMs["Exp"] == exp) & (SBMs["Run"] == run) &
			     (SBMs["Level"] == int(level))]
			for exp, runs in levels.items() for run, level in runs.items()
		]
		return pd.concat(rows, ignore_index=True) if rows else SBMs.iloc[0:0].copy()

	# Topic models have a native clustering; embedding methods are evaluated by K-means.
	methods = [
		_load_method_results(dataset, "Mowgli", label, "Mowgli"),
		_load_method_results(dataset, "MOFA", label, "MOFA+", embedding_clustering),
		_load_method_results(dataset, "COBOLT", label, "COBOLT"),
	]
	if dataset in ["BMMCCite", "Spleen"]:
		methods.append(_load_method_results(
			dataset, "TotalVI", label, "totalVI", embedding_clustering
		))
	else:
		methods += [
			_load_method_results(dataset, "ShareTopic", label, "ShareTopic"),
			_load_method_results(dataset, "MultiVI", label, "MultiVI", embedding_clustering),
		]

	other = pd.concat([x for x in methods if not x.empty], ignore_index=True, sort=False)
	df_best = pd.concat([select_levels(best), other], ignore_index=True, sort=False)
	df_ct = pd.concat([select_levels(closest), other], ignore_index=True, sort=False)
	for df in (df_best, df_ct):
		df["Data"] = df["Data"].astype(str).str.replace("_", "+", regex=False)
	return df_best, df_ct

def create_results_topics(dataset, exps=None):
	
	SBMs=pd.read_csv(f"Datasets/{dataset}/bionSBM/{dataset}_bionSBM_TopicsSpecDist.tsv.gz", sep="\t", index_col=0)
	SBMs["Algorithm"]="bionSBM"
	SBMs["Data"]=[create_data(SBMs.iloc[i]["Exp"]) for i in range(len(SBMs))]   
  
	mowgli=pd.read_csv(f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_TopicsSpecDist.tsv.gz", sep="\t", index_col=0)
	mowgli["Algorithm"]="Mowgli"
	mowgli["Data"]=["_".join(mowgli.iloc[i]["Exp"].split("_")[1:]) for i in range(len(mowgli))]	
	mowgli["FS"]="Mixed"
		
	if dataset not in ["BMMCCite", "Spleen"]:
		ShareTopic=pd.read_csv(f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_TopicsSpecDist.tsv.gz", sep="\t", index_col=0)
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



def _fresh_process_peak_worker(conn, fn, args, kwargs, poll_s, include_children, discard_result):
    """Run one measurement inside a newly spawned Python process."""
    import traceback

    try:
        result, elapsed, peak_increase = run_with_peak_increase(
            fn,
            *args,
            poll_s=poll_s,
            include_children=include_children,
            discard_result=discard_result,
            **kwargs,
        )
        conn.send(("ok", result, elapsed, peak_increase))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def run_with_peak_increase_fresh_process(
    fn,
    *args,
    poll_s: float = 0.01,
    include_children: bool = True,
    discard_result: bool = False,
    **kwargs,
):
    """
    Measure incremental RAM in a fresh spawned Python process.

    The child receives/unpickles the input first. Its RSS baseline is then sampled
    immediately before ``fn`` starts, so imports and the input object are excluded
    from the reported peak increase. A new child is used for every call, preventing
    allocator state from previous benchmark sizes from affecting the measurement.

    Returns
    -------
    (result_or_None, time_sec, peak_increase_bytes)
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    process = ctx.Process(
        target=_fresh_process_peak_worker,
        args=(
            send_conn,
            fn,
            args,
            kwargs,
            poll_s,
            include_children,
            discard_result,
        ),
    )
    process.start()
    send_conn.close()

    try:
        message = recv_conn.recv()
    except EOFError as exc:
        process.join()
        raise RuntimeError(
            f"Fresh benchmark process exited with code {process.exitcode} "
            "before returning a measurement"
        ) from exc
    finally:
        recv_conn.close()

    process.join()

    if message[0] == "error":
        raise RuntimeError("Fresh benchmark process failed:\n" + message[1])

    if process.exitcode != 0:
        raise RuntimeError(
            f"Fresh benchmark process exited with code {process.exitcode}"
        )

    _, result, elapsed, peak_increase = message
    return result, elapsed, peak_increase



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


# ============================================================================
# FEATURE-MATCHED REGULATORY NULL MODELS
# ============================================================================
# Shared by run_TF_enrichment_corrected_v2.py and
# run_TF_ATAC_GEX_corrected_v2.py. Random mRNA topics preserve the observed
# joint distribution of mean expression, cell coverage, and genomic gene length.

import gzip
import re

# Dataset-specific gene annotations. Paths are relative to the directory from
# which the analysis scripts are launched.
GTF_PATHS = {
    "BMMCCite": "../AnnotRef/hs/hg38/gencode.v44.annotation.gtf.gz",
    "BMMCMultiOme": "../AnnotRef/hs/hg38/gencode.v44.annotation.gtf.gz",
    "HSPC": "../AnnotRef/hs/hg38/gencode.v44.annotation.gtf.gz",
    "MouseSkin": "../AnnotRef/mm/gencode.vM33.annotation.gtf.gz",
    "PBMC": "../AnnotRef/hs/T2T/GCF_009914755.1_T2T-CHM13v2.0_genomic.gtf.gz",
    "Spleen": "../AnnotRef/mm/gencode.vM33.annotation.gtf.gz",
}

MOUSE_DATASETS = {"MouseSkin", "Spleen"}
MATCHING_COVARIATES = ("mean_expression", "coverage", "gene_length")


def strip_gene_version(name):
    """Remove a terminal Ensembl version: ENSG...12.4 -> ENSG...12."""
    return re.sub(r"\.\d+$", "", str(name))


def regulatory_matrix_path(dataset):
    """Return the fixed MuData file used to compute expression and coverage."""
    main_fs = "Peak" if dataset not in {"BMMCCite", "Spleen"} else "ADT"
    return f"Datasets/{dataset}/CM/{dataset}_{main_fs}_mRNA_lncRNA_Def.h5mu"


def _get_mrna_modality(mdata):
    """Return the mRNA/GEX modality without assuming exact capitalization."""
    keys = list(mdata.mod.keys())
    exact = {"mrna", "gex", "gene_expression", "gene expression", "rna"}
    for key in keys:
        if key.lower() in exact:
            return mdata.mod[key]
    for key in keys:
        if "mrna" in key.lower() or "gex" in key.lower():
            return mdata.mod[key]
    raise KeyError(f"Cannot identify mRNA modality; available modalities: {keys}")


def read_expression_lookup(dataset):
    """Map gene aliases to mean expression and cell coverage from ``adata.X``.

    Coverage is the fraction of cells with a strictly positive value. Sparse
    matrices are processed without conversion to a dense matrix.
    """
    path = regulatory_matrix_path(dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    adata = _get_mrna_modality(mu.read_h5mu(path))
    X = adata.X.copy() if sp.issparse(adata.X) else np.asarray(adata.X)
    if sp.issparse(X):
        X.eliminate_zeros()
        means = np.asarray(X.mean(axis=0)).ravel()
        coverage = np.asarray(X.getnnz(axis=0)).ravel() / adata.n_obs
    else:
        means = X.mean(axis=0)
        coverage = (X > 0).mean(axis=0)

    alias_columns = [c for c in (
        "gene_id", "gene_ids", "gene_name", "gene_names",
        "gene_symbol", "gene_symbols", "symbol"
    ) if c in adata.var.columns]

    lookup = {}
    for i, var_name in enumerate(adata.var_names.astype(str)):
        aliases = [var_name] + [adata.var.iloc[i][c] for c in alias_columns]
        for alias in aliases:
            if pd.isna(alias):
                continue
            for key in {str(alias), strip_gene_version(alias)}:
                lookup.setdefault(key, (float(means[i]), float(coverage[i])))
    return lookup


def _parse_gtf_attributes(text):
    """Parse quoted GTF attributes, with a small ``key=value`` fallback."""
    attrs = dict(re.findall(r'(\S+)\s+"([^"]*)"', text))
    if not attrs:
        attrs = dict(part.split("=", 1) for part in text.split(";") if "=" in part)
    return attrs


def read_gtf_lookup(path):
    """Map ``gene_id`` and ``gene_name`` aliases to genomic gene length.

    Only records whose third GTF column is ``gene`` are used. Length is the
    inclusive genomic span ``end - start + 1``. Both plain and gzipped GTFs are
    supported, including GENCODE and NCBI/T2T attribute layouts.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Invalid GTF path: {path}")

    lookup = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = _parse_gtf_attributes(fields[8])
            gene_id, gene_name = attrs.get("gene_id"), attrs.get("gene_name")
            if not gene_id:
                continue
            length = int(fields[4]) - int(fields[3]) + 1
            record = (str(gene_id), str(gene_name) if gene_name else None, length)
            for alias in (gene_id, strip_gene_version(gene_id), gene_name):
                if alias:
                    lookup.setdefault(str(alias), record)
    if not lookup:
        raise ValueError(f"No gene records found in GTF: {path}")
    return lookup


def read_gene_lookups(dataset):
    """Load expression/coverage and GTF-length lookup tables once."""
    if dataset not in GTF_PATHS:
        raise KeyError(f"No GTF configured for dataset: {dataset}")
    return read_expression_lookup(dataset), read_gtf_lookup(GTF_PATHS[dataset])


def gene_covariates(features, expression_lookup, gtf_lookup):
    """Resolve topic genes and return their three matching covariates.

    Topic features are searched as provided, without Ensembl versions, and via
    both ``gene_id`` and ``gene_name`` aliases from the GTF. Genes missing any
    covariate are excluded because they cannot be feature-matched.
    """
    rows = []
    for feature in map(str, features):
        keys = [feature, strip_gene_version(feature)]
        record = next((gtf_lookup[k] for k in keys if k in gtf_lookup), None)
        if record:
            gene_id, gene_name, length = record
            keys += [gene_id, strip_gene_version(gene_id), gene_name]
        else:
            length = np.nan
        expression, coverage = next(
            (expression_lookup[k] for k in keys if k and k in expression_lookup),
            (np.nan, np.nan),
        )
        rows.append((feature, expression, coverage, length))

    stats = pd.DataFrame(
        rows, columns=("gene",) + MATCHING_COVARIATES
    ).set_index("gene")
    return stats.replace([np.inf, -np.inf], np.nan).dropna(
        subset=MATCHING_COVARIATES
    )


def add_matching_strata(stats, n_bins=5):
    """Assign joint quantile bins for expression, coverage, and gene length."""
    stats = stats.copy()
    labels = []
    for covariate in MATCHING_COVARIATES:
        values = stats[covariate]
        if values.nunique() <= 1:
            binned = pd.Series(0, index=values.index, dtype=int)
        else:
            binned = pd.qcut(
                values, min(n_bins, values.nunique()), labels=False,
                duplicates="drop",
            ).astype(int)
        labels.append(binned.rename(covariate))
    stats["matching_stratum"] = (
        pd.concat(labels, axis=1).astype(str).agg("|".join, axis=1)
    )
    return stats


def iter_matched_null_topics(topic_genes, stats, n_perm, rng):
    """Yield same-size random topics with identical joint stratum counts.

    Observed topic genes are excluded from a stratum when enough alternative
    genes exist. In rare small strata, the complete stratum is used to preserve
    exact topic size and matching distributions.
    """
    topic = set(map(str, topic_genes)) & set(stats.index)
    if not topic:
        return

    counts = stats.loc[list(topic), "matching_stratum"].value_counts()
    pools = {}
    for stratum, size in counts.items():
        mask = (stats["matching_stratum"] == stratum) & ~stats.index.isin(topic)
        pool = stats.index[mask].to_numpy()
        if len(pool) < size:
            pool = stats.index[stats["matching_stratum"] == stratum].to_numpy()
        pools[stratum] = pool

    for _ in range(n_perm):
        random_topic = set()
        for stratum, size in counts.items():
            random_topic.update(
                map(str, rng.choice(pools[stratum], int(size), replace=False))
            )
        yield random_topic


def find_best_topic_level(base_dir, n_celltypes):
    """Find the hierarchy level with cluster count closest to cell-type count.

    This name is intentionally distinct from the pre-existing ``find_best_level``
    utility above, whose API is used by the benchmarking code.
    """
    best_level, best_delta = None, np.inf
    for name in os.listdir(base_dir):
        if not name.endswith("_clusters.tsv.gz") or "_level_" not in name:
            continue
        try:
            clusters = pd.read_csv(
                os.path.join(base_dir, name), sep="\t", index_col=0
            )
        except Exception:
            continue
        n_clusters = np.unique(np.argmax(clusters.values, axis=0)).size
        delta = abs(n_clusters - n_celltypes)
        if delta < best_delta:
            best_level = name.split("_level_")[1].split("_")[0]
            best_delta = delta
    return best_level


def load_topic_gene_sets(path):
    """Load an mRNA topic matrix and return its nonzero gene sets."""
    topics = pd.read_csv(path, sep="\t", index_col=0)
    topics.index = topics.index.astype(str).str.replace("#", "", regex=False)
    topics.replace(0, np.nan, inplace=True)
    gene_sets = {topic: set(topics[topic].dropna().index) for topic in topics.columns}
    return topics, gene_sets


def load_regulons(dataset):
    """Load CollecTRI and return TF-to-target-gene sets."""
    path = "collectri_mm.tsv" if dataset in MOUSE_DATASETS else "collectri.tsv"
    network = pd.read_csv(path, sep="\t")
    return network.groupby("source")["target"].apply(set).to_dict()


def bh_fdr(pvalues):
    """Return Benjamini-Hochberg adjusted P-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    if not len(pvalues):
        return pvalues
    order = np.argsort(pvalues)
    adjusted = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output
  


def read_celltype(count_matrix):
	mdata = mu.read_h5mu(count_matrix)

	if "CellType" in mdata.obs.columns:
		return mdata.obs["CellType"].copy()

	for mod in mdata.mod:
		if "CellType" in mdata[mod].obs.columns:
			return mdata[mod].obs["CellType"].copy()

	raise KeyError(f"No CellType column found in {count_matrix}")


def align_labels(metadata, clusters):
	metadata = metadata.dropna()
	clusters = clusters.dropna()

	cells = metadata.index.intersection(clusters.index, sort=False)
	return metadata.loc[cells].astype(str), clusters.loc[cells].astype(str)


def factor_labels(file_data, absolute=False):
	"""
	Read a dimensions x cells matrix and assign each cell to its
	largest latent factor.
	"""
	embedding = pd.read_csv(file_data, index_col=0, sep="\t")

	if absolute:
		embedding = embedding.abs()

	return pd.Series(
		np.argmax(embedding.to_numpy(), axis=0),
		index=embedding.columns,
		name="cluster",
	)


def kmeans_labels(file_data, metadata, run):
	"""
	Read a dimensions x cells embedding and cluster cells using
	K-means with k equal to the number of ground-truth cell types.
	"""
	embedding = pd.read_csv(file_data, index_col=0, sep="\t").T
	embedding = embedding.loc[~embedding.index.duplicated()]
	embedding = embedding.replace([np.inf, -np.inf], np.nan).dropna()

	cells = embedding.index.intersection(metadata.dropna().index, sort=False)
	embedding = embedding.loc[cells]
	y_true = metadata.loc[cells]

	n_clusters = y_true.nunique()
	if n_clusters < 2 or n_clusters >= len(embedding):
		return pd.Series(dtype=str, name="cluster")

	x = StandardScaler().fit_transform(embedding)

	clusters = KMeans(
		n_clusters=n_clusters,
		n_init=20,
		random_state=run,
	).fit_predict(x)

	return pd.Series(
		clusters,
		index=embedding.index,
		name="cluster",
	)


def topic_labels(file_data):
	"""
	Read a one-column MaxTopic file.
	"""
	data = pd.read_csv(file_data, index_col=0, sep="\t")

	if data.shape[1] != 1:
		raise ValueError(
			f"Expected one cluster column in {file_data}, "
			f"found {data.shape[1]}"
		)

	return data.iloc[:, 0].rename("cluster")


def compute_nmi(metadata, clusters, seed, n_shuffles=100):
	y_true, y_pred = align_labels(metadata, clusters)

	if len(y_true) == 0 or y_pred.nunique() < 2:
		return np.nan, np.nan, 0

	observed = nmi.compute_normalised_mutual_information(
		y_true,
		y_pred,
	)

	rng = np.random.default_rng(seed)
	random_nmi = np.mean([
		nmi.compute_normalised_mutual_information(
			y_true,
			rng.permutation(y_pred.to_numpy()),
		)
		for _ in range(n_shuffles)
	])

	return observed, random_nmi, y_pred.nunique()
