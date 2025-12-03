import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import os, time, threading, gc, psutil, sklearn
import mowgli
from pathlib import Path
from helps import *

mdata=mu.read_h5mu("Datasets/HSPC/CM/HSPC_Peak_mRNA_lncRNA_Def.h5mu")
mdata

sizes=np.linspace(500, mdata.shape[0], 10).astype(int)

def run_mowgli(obj):
	adatas={}
	for mod in obj.mod:
		adata=obj[mod].copy()
		adata.X=adata.X.toarray().copy()
		adatas[mod]=adata
	obj=mu.MuData(adatas)
	
	h_regularization={mod : 5e-2 for mod in list(obj.mod.keys())}
	model = mowgli.models.MowgliModel(latent_dim=len(set(obj.obs["Peak:CellType"].dropna())), h_regularization=h_regularization)
	model.train(obj, device="cpu")
	Path("TEMPMowgli").mkdir(parents=True, exist_ok=True)
	embeddding=pd.DataFrame(obj.obsm["W_OT"].T, columns=obj.obs.index, index=[f"Dim_{i}" for i in range(0, model.latent_dim)])
	embeddding.to_csv("TEMPMowgli/Embedding.tsv.gz", compression="gzip", sep="\t")
	os.system("rm -rf TEMPMowgli")

# ---------- Benchmark loop ----------
rows = []
for n_cells in sizes:
	print(f"Starting with {n_cells}", flush=True)

	subsample_dict = {'cells': n_cells}
	mdata_sub = muon_subsample(mdata, subsample_dict, strata_column="CellType")

	_, t_mowgli, ram_mowgli = run_with_peak_increase(run_mowgli, mdata_sub, discard_result=True)
	
	rows.append({"n_cells": n_cells, "t_mowgli": t_mowgli, "ram_mowgli": ram_mowgli / 1e9})

	print(f"[{n_cells:>9} cells] | time: {t_mowgli:.2f}s | RAM: {ram_mowgli/1e9:.2f}GB", flush=True)

	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_Mowgli_cells.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_Mowgli_cells.tsv", flush=True)

rows=[]
keys=[["Peak"],["Peak","mRNA"],["Peak","mRNA","lncRNA"]]
for key in keys:
	new_mdata=mu.MuData({k : mdata.mod[k] for k in key})
	print(f"Starting with {len(key)} modalities", flush=True)
	subsample_dict = {'cells': 5000} | {k : 1000 for k in key}
	mdata_sub = muon_subsample(new_mdata, subsample_dict, strata_column="CellType")
	
	_, t_mowgli, ram_mowgli = run_with_peak_increase(run_mowgli, mdata_sub, discard_result=True)  
	
	rows.append({"N_modalities": len(key),"t_mowgli": t_mowgli,"ram_mowgli": ram_mowgli / 1e9})
	
	print(f"[{len(key):>9} modalities] | time: {t_mowgli:.2f}s | RAM: {ram_mowgli/1e9:.2f}GB", flush=True)
	
	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_Mowgli_modalities.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_owgli_modalities.tsv", flush=True)
