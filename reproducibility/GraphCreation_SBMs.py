import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import bionsbm
import os, time, threading, gc, psutil, sklearn
from helps import *
from nsbm import nsbm

mdata=mu.read_h5mu("Datasets/HSPC/CM/HSPC_Peak_mRNA_lncRNA_Def.h5mu")
mdata

sizes=np.linspace(500, mdata.shape[0], 10).astype(int)

def run_bionsbm(obj):
	model = bionsbm.model.bionsbm(obj=obj, load_if_exists=False)
	os.system("rm -rf results")

def run_nsbm(obj):
	model2 = nsbm()
	model2.make_graph_multiple_df(obj["Peak"].to_df().T, [obj[key].to_df().T for key in list(obj.mod.keys())[1:]])
	os.system("rm -rf results")

# ---------- Benchmark loop ----------
rows = []
for n_cells in sizes:
	print(f"Starting with {n_cells}", flush=True)

	subsample_dict = {'cells': n_cells}
	mdata_sub = muon_subsample(mdata, subsample_dict, strata_column="CellType")

	_, t_bionsbm, ram_bionsbm = run_with_peak_increase(run_bionsbm, mdata_sub, discard_result=True)
	_, t_nsbm, ram_nsbm = run_with_peak_increase(run_nsbm, mdata_sub, discard_result=True)
	
	rows.append({"n_cells": n_cells, "t_bionsbm": t_bionsbm, "ram_bionsbm": ram_bionsbm / 1e9, 
				 					 "t_nsbm": t_nsbm, "ram_nsbm": ram_nsbm / 1e9})

	print(f"[{n_cells:>9} cells] | Bionsbm--> time: {t_bionsbm:.2f}s | RAM: {ram_bionsbm/1e9:.2f}GB \n nSBM --> time: {t_nsbm:.2f}s | RAM: {ram_nsbm/1e9:.2f}GB", flush=True)

	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_bionSBM_vs_nSBM_cells.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_bionSBM_vs_nSBM_cells.tsv", flush=True)
