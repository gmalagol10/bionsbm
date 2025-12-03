import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import muon as mu
import pandas as pd

import torch
import mowgli
import time
import sys
import os

from pathlib import Path

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
print(f'Running on: {device}')

print(f"Mowgli script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

cm=sys.argv[1]
path_to_save=sys.argv[2]

folder=os.path.dirname(path_to_save)
name=path_to_save.split("/")[-1]
Path(folder).mkdir(parents=True, exist_ok=True)

names=["Script", "CM", "Path to save"]
for nm,arg in zip(names,sys.argv):
	print(nm,":", arg, flush=True)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Reading count matrix", flush=True)
mdata=mu.read_h5mu(cm)
adatas={}
for mod in mdata.mod:
    adata=mdata[mod].copy()
    adata.X=adata.X.toarray().copy()
    adatas[mod]=adata
mdata=mu.MuData(adatas)
del adatas

h_regularization={mod : 5e-2 for mod in list(mdata.mod.keys())}

for run in range(0, 10):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/{runs}", flush=True)
	model = mowgli.models.MowgliModel(latent_dim=len(set(mdata.obs[f"{mod}:CellType"].dropna())), h_regularization=h_regularization)

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/{runs} Fitting Mowgli model", flush=True)
	model.train(mdata, device=device)

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/{runs} Saving results", flush=True)
	Path(f"{folder}/Runs/Run{run}").mkdir(parents=True, exist_ok=True)
	embeddding=pd.DataFrame(mdata.obsm["W_OT"].T, columns=mdata.obs.index, index=[f"Dim_{i}" for i in range(0, model.latent_dim)])
	embeddding.to_csv(f"{folder}/Runs/Run{run}/{name}_Embedding.tsv.gz", compression="gzip", sep="\t")
