import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import os, time, threading, gc, psutil, sklearn
from SHARE_topic import SHARE_topic
from pathlib import Path
from helps import *

mdata=mu.read_h5mu("Datasets/HSPC/CM/HSPC_Peak_mRNA_lncRNA_Def.h5mu")
sizes=np.linspace(500, mdata.shape[0], 10).astype(int)

def run_sharetopic(obj):
	key1=list(obj.mod.keys())[0]
	key2=list(obj.mod.keys())[1]
	obj[key1].X=obj[key1].layers["raw"].copy()
	obj[key2].X=obj[key2].layers["raw"].copy()

	gamma=1
	tau=0.5
	n_topics=len(set(obj[key1].obs.CellType.dropna()))
	n_samples=100
	n_burnin=1
	batch_size=50
	alpha=50/n_topics
	beta=0.1
	st_obj = SHARE_topic(obj[key1], obj[key2], n_topics, alpha, beta, gamma, tau)
	theta, lam, phi = st_obj.fit(batch_size,n_samples,n_burnin,dev="cpu",save_data=False)
	waic = st_obj.WAIC(batch_size, theta[0:,:,:], lam[0:,:,:], phi[0:,:,:], "cpu")

	m_theta = theta[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_theta = m_theta/m_theta.sum(axis=1)[:,np.newaxis] 

	m_phi = phi[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_phi = m_phi/m_phi.sum(axis=1)[:,np.newaxis] 

	m_lam = lam[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_lam = m_lam/m_lam.sum(axis=1)[:,np.newaxis] 

	top_cell=pd.DataFrame(m_theta.cpu().detach().numpy(), index=obj[key2].obs.index, columns=[f"Topic_{t}" for t in range(0, n_topics)]).T
	top_key1=pd.DataFrame(m_phi.cpu().detach().numpy(), index=[f"Topic_{t}" for t in range(0, n_topics)], columns=obj[key1].var.iloc[st_obj.region_id_lookup])
	top_key2=pd.DataFrame(m_lam.cpu().detach().numpy(), index=[f"Topic_{t}" for t in range(0, n_topics)], columns=obj[key2].var.index)
	d=pd.DataFrame(list(np.argmax(top_cell.values, axis=0).astype(str)), columns=["BestTopic"], index=obj[key1].obs.index)

	Path("TempShare").mkdir(parents=True, exist_ok=True)
	top_cell.to_csv("TempShare/todelete_Topic_Cell.tsv.gz", compression="gzip", sep="\t")
	top_key1.to_csv(f"TempShare/todelete_Topic_{key1}.tsv.gz", compression="gzip", sep="\t")
	top_key2.to_csv(f"TempShare/todelete_Topic_{key2}.tsv.gz", compression="gzip", sep="\t")
	d.to_csv("TempShare/todelete_MaxTopic.tsv.gz", compression="gzip", sep="\t")

	os.system("rm -rf TempShare")

# ---------- Benchmark loop ----------
rows = []
for n_cells in sizes:
	print(f"Starting with {n_cells}", flush=True)

	subsample_dict = {'cells': n_cells}
	mdata_sub = muon_subsample(mdata, subsample_dict, strata_column="CellType")

	_, t_share, ram_share = run_with_peak_increase(run_sharetopic, mdata_sub, discard_result=True)
	
	rows.append({"n_cells": n_cells, "t_sharetopic": t_share, "ram_sharetopic": ram_share / 1e9})

	print(f"[{n_cells:>9} cells] | time: {t_share:.2f}s | RAM: {ram_share/1e9:.2f}GB", flush=True)

	# Save results
	df = pd.DataFrame(rows)
	df.to_csv("Tables/Time_ShareTopic_cells.tsv", sep="\t", index=False)
	print("Saved: Tables/Time_ShareTopic_cells.tsv", flush=True)
