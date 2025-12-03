import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import bionsbm
import os, time, threading, gc, psutil, sklearn
from helps import *

mdata=mu.read_h5mu("Datasets/HSPC/CM/HSPC_Peak_mRNA_lncRNA_Def.h5mu")
mdata

sizes=np.linspace(500, mdata.shape[0], 10).astype(int)

def run_bionsbm(obj):
	model = bionsbm.model.bionsbm(obj=obj, load_if_exists=False)
	model.fit(n_init=1, verbose=False)
	os.system("rm -rf results")

# ---------- Benchmark loop ----------
rows = []
for n_cells in sizes:
	print(f"Starting with {n_cells}", flush=True)

	subsample_dict = {'cells': n_cells}
	mdata_sub = muon_subsample(mdata, subsample_dict, strata_column="CellType")

	_, t_bionsbm, ram_bionsbm = run_with_peak_increase(run_bionsbm, mdata_sub, discard_result=True)
	
	rows.append({"n_cells": n_cells, "t_bionsbm": t_bionsbm, "ram_bionsbm": ram_bionsbm / 1e9})

	print(f"[{n_cells:>9} cells] | time: {t_bionsbm:.2f}s | RAM: {ram_bionsbm/1e9:.2f}GB", flush=True)

	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_bionSBM_cells.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_bionSBM_cells.tsv", flush=True)



rows=[]
keys=[["Peak"],["Peak","mRNA"],["Peak","mRNA","lncRNA"]]
for key in keys:
	new_mdata=mu.MuData({k : mdata.mod[k] for k in key})
	print(f"Starting with {len(key)} modalities", flush=True)
	subsample_dict = {'cells': 5000} | {k : 1000 for k in key}
	mdata_sub = muon_subsample(new_mdata, subsample_dict, strata_column="CellType")

	_, t_bionsbm, ram_bionsbm = run_with_peak_increase(run_bionsbm, mdata_sub, discard_result=True)  
		
	rows.append({"N_modalities": len(key),"t_bionsbm": t_bionsbm,"ram_bionsbm": ram_bionsbm / 1e9})
	
	print(f"[{len(key):>9} modalities] | time: {t_bionsbm:.2f}s | RAM: {ram_bionsbm/1e9:.2f}GB", flush=True)
	
	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_bionSBM_modalities.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_bionSBM_modalities.tsv", flush=True)
