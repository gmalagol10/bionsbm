import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import os
import sys
import time

#+++++++++++++++++++++++++++++++ bionSBM +++++++++++++++++++++++++++++++
for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC","BMMCCite","Spleen"]:
	meta=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
	data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
	if dataset in ["BMMCCite", "Spleen"]:
		exps=["bionSBM_ADT","bionSBM_mRNA","bionSBM_ADT_GEX", "bionSBM_ADT_mRNA", "bionSBM_ADT_lncRNA", "bionSBM_ADT_mRNA_lncRNA"]
	else:
		exps=["bionSBM_Peak","bionSBM_mRNA","bionSBM_Peak_GEX", "bionSBM_Peak_mRNA", "bionSBM_Peak_lncRNA", "bionSBM_Peak_mRNA_lncRNA"]
	for exp in exps:
		fss = exp.split("_")[1:]
		for run in range(0, 25):
			for fs in fss:
				levels={}
				for level in range(6):
					file = f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/{exp}_level_{level}_{fs}_topics_documents.tsv.gz"
					if os.path.isfile(file):
						levels[level] = np.abs(pd.read_csv(file, sep="\t", nrows=0).shape[1]-len(meta["CellType"].unique()))
				if len(levels.keys()) > 0:
					level = min(levels, key=levels.get)
					file = f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/{exp}_level_{level}_{fs}_topics_documents.tsv.gz"
					if os.path.isfile(file):
						df=pd.read_csv(file, index_col=0, sep="\t")
						if df.shape[1] > 0:
							df=df-df.mean(axis=0)
							df["CT"]=meta.loc[df.index]["CellType"]
							df=df.groupby("CT").mean()
							df=(df-df.min())/(df.max()-df.min()) 
							diff=len(set(df.idxmax(axis=1)))
							for ct, tp in dict(df.idxmax(axis=1).dropna()).items():
								cols=list(df.columns)
								cols.remove(tp)
								d=pd.DataFrame(data=[dataset,exp,run,fs,ct,diff, df.shape[0], df.shape[1], df.loc[ct][tp]-df.loc[ct][cols].mean()], index=data_def.columns).T
								data_def=pd.concat([data_def,d],axis=0)
					else:
						print("File", file,"not found", flush=True)
					print("No levels in", dataset, exp, run, fs, flush=True)
	data_def.to_csv(f"Datasets/{dataset}/bionSBM/{dataset}_bionSBM_TopicsSpecDist.tsv.gz", sep="\t", compression="gzip")

#+++++++++++++++++++++++++++++++ ShareTopic +++++++++++++++++++++++++++++++
for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC"]:
	meta=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
	data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
	exps=["ShareTopic_Peak_GEX", "ShareTopic_Peak_mRNA", "ShareTopic_Peak_lncRNA"]
	for exp in exps:
		for run in range(0, 25):
			file = f"Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
			if os.path.isfile(file):
				df=pd.read_csv(file, index_col=0, sep="\t").T
				df=df-df.mean(axis=0)
				df["CT"]=meta.loc[df.index]["CellType"]
				df=df.groupby("CT").mean()
				df=(df-df.min())/(df.max()-df.min()) 
				diff=len(set(df.idxmax(axis=1)))
				for ct, tp in dict(df.idxmax(axis=1).dropna()).items():
					cols=list(df.columns)
					cols.remove(tp)
					d=pd.DataFrame(data=[dataset,exp,run,fs,ct,diff, df.shape[0], df.shape[1], df.loc[ct][tp]-df.loc[ct][cols].mean()], index=data_def.columns).T
					data_def=pd.concat([data_def,d],axis=0)
	data_def.to_csv(f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_TopicsSpecDist.tsv.gz", sep="\t", compression="gzip")

#+++++++++++++++++++++++++++++++ Mowgli +++++++++++++++++++++++++++++++
for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC","BMMCCite","Spleen"]:
	meta=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
	data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
	if dataset in ["BMMCCite", "Spleen"]:
		exps = ["Mowgli_ADT_GEX", "Mowgli_ADT_mRNA", "Mowgli_ADT_lncRNA", "Mowgli_ADT_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
	else:
		exps = ["Mowgli_Peak_GEX", "Mowgli_Peak_mRNA", "Mowgli_Peak_lncRNA", "Mowgli_Peak_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
	for exp in exps:
		for run in range(0, 25):
			file = f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
			if os.path.isfile(file):
				df=pd.read_csv(file, index_col=0, sep="\t").T
				df=df-df.mean(axis=0)
				df["CT"]=meta.loc[df.index]["CellType"]
				df=df.groupby("CT").mean()
				df=(df-df.min())/(df.max()-df.min()) 
				diff=len(set(df.idxmax(axis=1)))
				for ct, tp in dict(df.idxmax(axis=1).dropna()).items():
					cols=list(df.columns)
					cols.remove(tp)
					d=pd.DataFrame(data=[dataset,exp,run,fs,ct,diff, df.shape[0], df.shape[1], df.loc[ct][tp]-df.loc[ct][cols].mean()], index=data_def.columns).T
					data_def=pd.concat([data_def,d],axis=0)
			else:
				print("File", file,"not found", flush=True)
	data_def.to_csv(f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_TopicsSpecDist.tsv.gz", sep="\t", compression="gzip")
