import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from sklearn.metrics import adjusted_mutual_info_score as AMI
from pathlib import Path
from scipy import io
from helps import *
import nmi

for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC","BMMCCite","Spleen"]:
    data=pd.read_csv(f"../Datasets/{dataset}/SBM/{dataset}_SBMs_25Run.tsv.gz", sep="\t", index_col=0)
    meta=pd.read_csv(f"../Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
    data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
    if dataset in ["BMMCCite", "Spleen"]:
    	exps_nSBM = ["nSBM_ADT_GEX", "nSBM_ADT_mRNA", "nSBM_ADT_lncRNA", "nSBM_ADT_mRNA_lncRNA", "nSBM_mRNA_lncRNA"]
    else:
    	exps_nSBM = ["nSBM_Peak_GEX", "nSBM_Peak_mRNA", "nSBM_Peak_lncRNA", "nSBM_Peak_mRNA_lncRNA", "nSBM_mRNA_lncRNA"]
    for exp in exps_nSBM:
        print(dataset, exp)
        fss = exp.split("_")[1:]
        for run in range(0, 25):
            exps_levels={}
            for level in range(0,4):
                if f"{exp}_Level_{level}_Run_{run}" in data.columns:
                    exps_levels[level]=np.abs(len(set(data[f"{exp}_Level_{level}_Run_{run}"].dropna()))-len(set(data["CellType"].dropna())))
            if bool(exps_levels) != False:
                min_level=np.argmin(pd.DataFrame.from_dict(exps_levels, orient="index").fillna(10000000000))
                for fs in fss:
                    file = f"../Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{min_level}_{fs}_topics_documents.csv.gz"
                    if os.path.isfile(file):
                        df=pd.read_csv(file, index_col=0)
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
                       print("File", file,"not found")
#    data_def.to_csv(f"../Datasets/{dataset}/SBM/{dataset}_SBMs_25Run_TopicsSpec.tsv.gz", sep="\t", compression="gzip")

for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC","BMMCCite","Spleen"]:
    meta=pd.read_csv(f"../Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
    data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
    if dataset in ["BMMCCite", "Spleen"]:
    	exps = ["Mowgli_ADT_GEX", "Mowgli_ADT_mRNA", "Mowgli_ADT_lncRNA", "Mowgli_ADT_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
    else:
    	exps = ["Mowgli_Peak_GEX", "Mowgli_Peak_mRNA", "Mowgli_Peak_lncRNA", "Mowgli_Peak_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
    for exp in exps:
        print(dataset, exp)
        for run in range(0, 25):
            file = f"../Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
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
 #   data_def.to_csv(f"../Datasets/{dataset}/Mowgli/{dataset}_Mowgli_25Run_TopicsSpec.tsv.gz", sep="\t", compression="gzip")


for dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC"]:
    meta=pd.read_csv(f"../Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
    data_def = pd.DataFrame(columns=["Dataset","Exp","Run","FS","CT","DifferentTopics","NumberOfCt","NumberOfTopics","Topics_distinctiveness"])
    exps=["ShareTopic_Peak_GEX", "ShareTopic_Peak_mRNA", "ShareTopic_Peak_lncRNA"]
    for exp in exps:
        print(dataset, exp)
        for run in range(0, 25):
            file = f"../Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
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
                    d=pd.DataFrame(data=[dataset,exp,run,fs,ct,diff, df.shape[0], df.shape[1], df.loc[ct][tp]-df.loc[ct][cols].mean()],
                                   index=data_def.columns).T
                    data_def=pd.concat([data_def,d],axis=0)
  #  data_def.to_csv(f"../Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_25Run_TopicsSpec.tsv.gz", sep="\t", compression="gzip")
