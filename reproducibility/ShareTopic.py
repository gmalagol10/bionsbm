import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import muon as mu

import os
import sys
import time
import time
import torch

from pathlib import Path
from SHARE_topic import SHARE_topic

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
print(f'Running on: {device}')

print(f"ShareTopic script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

cm=sys.argv[1]
path_to_save=sys.argv[2]

folder=os.path.dirname(path_to_save)
name=path_to_save.split("/")[-1]
Path(folder).mkdir(parents=True, exist_ok=True)

names=["Script", "CM", "Path to save"]
for nm,arg in zip(names,sys.argv):
	print(nm,":", arg, flush=True)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Reading AnnData objects:", flush=True)
mdata=mu.read_h5mu(cm)
key1=list(mdata.mod.keys())[0]
key2=list(mdata.mod.keys())[1]

gamma=1
tau=0.5
n_topics=len(set(mdata[key1].obs.CellType.dropna()))
n_samples=100
n_burnin=1
batch_size=50
alpha=50/n_topics
beta=0.1

for run in range(0, 25):
	Path(f"{folder}/Runs/Run{run}").mkdir(parents=True, exist_ok=True)

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Starting Share-Topic run {run}", flush=True)
	st_obj = SHARE_topic(mdata[key1], mdata[key2], n_topics, alpha, beta, gamma, tau)
	theta, lam, phi = st_obj.fit(batch_size,n_samples,n_burnin,dev= device,save_data=False)
	waic = st_obj.WAIC(batch_size, theta[0:,:,:], lam[0:,:,:], phi[0:,:,:], "cpu")

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Saving results run {run}", flush=True)
	m_theta = theta[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_theta = m_theta/m_theta.sum(axis=1)[:,np.newaxis] 

	m_phi = phi[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_phi = m_phi/m_phi.sum(axis=1)[:,np.newaxis] 

	m_lam = lam[0:,:,:][n_samples-1:,:,:].mean(axis=0)
	m_lam = m_lam/m_lam.sum(axis=1)[:,np.newaxis] 

	top_cell=pd.DataFrame(m_theta.cpu().detach().numpy(), index=mdata[key2].obs.index, columns=[f"Topic_{t}" for t in range(0, n_topics)]).T
	top_key1=pd.DataFrame(m_phi.cpu().detach().numpy(), index=[f"Topic_{t}" for t in range(0, n_topics)], columns=mdata[key1].var.index)
	top_key2=pd.DataFrame(m_lam.cpu().detach().numpy(), index=[f"Topic_{t}" for t in range(0, n_topics)], columns=mdata[key2].var.index)

	d=pd.DataFrame(list(np.argmax(top_cell.values, axis=0).astype(str)), columns=[f"Run{run}"], index=mdata[key1].obs.index)
	top_cell.to_csv(f"{folder}/Runs/Run{run}/{name}_Topic_Cell.tsv.gz", compression="gzip", sep="\t")
	top_key1.to_csv(f"{folder}/Runs/Run{run}/{name}_Topic_{key1}.tsv.gz", compression="gzip", sep="\t")
	top_key2.to_csv(f"{folder}/Runs/Run{run}/{name}_Topic_{key2}.tsv.gz", compression="gzip", sep="\t")
	d.to_csv(f"{folder}/Runs/Run{run}/{name}_MaxTopic.tsv.gz", compression="gzip", sep="\t")
