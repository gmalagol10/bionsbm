import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import glob
import os
import random
import sys
import json
import time

from helps import *
from pathlib import Path


dataset=sys.argv[1]
annots_path=sys.argv[2] # in the form like ../AnnotRef/hs/T2T/BEDFiles/T2T

names=["Script --> ","Dataset -->","Experiment -->","Annots path -->"]
for nm,arg in zip(names,sys.argv):
	print(nm,":", arg, flush=True)

exps=["hSBM_Peak","nSBM_Peak_GEX", "nSBM_Peak_mRNA", "nSBM_Peak_lncRNA", "nSBM_Peak_mRNA_lncRNA"]
annots=["lncRNA","promoter","protein_coding","miRNA","tRNA","rRNA","LINE","SINE"]

if os.path.isfile(f"Datasets/{dataset}/CM/{dataset}_Peak_Def_Annot.h5ad"):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),"Annotated peaks matrix found!", flush=True)
	adata=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_Peak_Def_Annot.h5ad")
else:
	adata=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_Peak_Def.h5ad")
	peaks=pd.DataFrame([pos.split("_") for pos in list(adata.var.index)])
	peaks.to_csv(f"Datasets/{dataset}/CM/{dataset}_Peaks.tsv", index=None, columns=None, header=None, sep="\t")
	os.system(f"sort -k1,1 -k2,2n -k3,3n Datasets/{dataset}/CM/{dataset}_Peaks.tsv > Datasets/{dataset}/CM/{dataset}_Peaks_Sorted.tsv")
	norms={}
	for annot in annots:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), annot, flush=True)
		os.system(f"bedtools intersect -wao -a Datasets/{dataset}/CM/{dataset}_Peaks_Sorted.tsv -b {annots_path}_{annot}.bed > Datasets/{dataset}/CM/{dataset}_Peaks_Sorted_{annot}.tsv")
		d=pd.read_csv(f"Datasets/{dataset}/CM/{dataset}_Peaks_Sorted_{annot}.tsv", sep="\t", header=None)
		d=d[d[d.columns[-1]]!=0]
		d.index=["_".join([str(d.iloc[i][0]),str(d.iloc[i][1]), str(d.iloc[i][2])]) for i in range(len(d))]
		adata.var[annot]=0
		for pos in intersection([d.index, adata.var.index]):
		    adata.var.at[pos, annot]=np.sum(d.loc[pos][d.columns[-1]])
		df=pd.read_csv(f"{annots_path}_{annot}.bed", sep="\t", header=None)
		norms[annot]=np.sum(df[2]-df[1])
	adata.uns["AnnotNorm"]=norms
	adata.write(f"Datasets/{dataset}/CM/{dataset}_Peak_Def_Annot.h5ad", compression="gzip")


exp=exps[0]
print(f"hSBM peaks", flush=True)
for run in range(0, 25):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Exp {exp} | Run -->", run, flush=True)
	for l in range(0,5):
		path=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}"
		file=f'{path}/{dataset}_{exp}_topics_level_{l}.txt'
		if os.path.isfile(file):
			with open(file) as f:
				d = json.load(f)
			d1={key : flat_list(np.array(d[key])[:,:1].tolist()).tolist() for key in d.keys()}
			temp=pd.DataFrame.from_dict(d1, orient="index").T
			if temp.shape[1] > 1 and temp.shape[1] < 500:
				topics={str(col) : pd.DataFrame(adata.var.loc[temp[col].dropna()][annots].sum(axis=0)).T.to_dict(orient="index")[0] for col in temp.columns}
				with open(f"{path}/Level_{l}_TopicsAnnots.json","w") as f:
					f.write(json.dumps(topics))


for exp in exps[1:]:
	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Exp {exp} | Run -->", run, flush=True)
		for l in range(0, 5):
			path=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}"
			file=f"{path}/{dataset}_{exp}_level_{l}_Peak _topics.csv.gz"
			if os.path.isfile(file):
				temp=pd.read_csv(file, index_col=0)
				if temp.shape[1] > 1 and temp.shape[1] < 500:
					topics={str(col) : pd.DataFrame(adata.var.loc[list(temp[col][temp[col]!=0].index)][annots].sum(axis=0)).T.to_dict(orient="index")[0] for col in temp.columns}
					with open(f"{path}/Level_{l}_TopicsAnnots.json","w") as f:
						f.write(json.dumps(topics))
