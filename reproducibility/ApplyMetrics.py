import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import glob
import os
import random
import sklearn
from sklearn.metrics import silhouette_score
import sys
import time
import nmi

from pathlib import Path


dataset=sys.argv[1]
print("Dataset:", dataset, flush=True)

#+++++++++++++++++++++++++++++++ bionSBM +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"bionSBM", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", index_col=0, sep="\t")
data=metadata.copy()
file_name=f"Datasets/{dataset}/bionSBM/{dataset}_bionSBM_clustering.tsv.gz"
if dataset in ["BMMCCite", "Spleen"]:
	exps=["bionSBM_ADT","bionSBM_mRNA","bionSBM_ADT_GEX", "bionSBM_ADT_mRNA", "bionSBM_ADT_lncRNA", "bionSBM_ADT_mRNA_lncRNA", "bionSBM_mRNA_lncRNA"]
else:
	exps=["bionSBM_Peak","bionSBM_mRNA","bionSBM_Peak_GEX", "bionSBM_Peak_mRNA", "bionSBM_Peak_lncRNA", "bionSBM_Peak_mRNA_lncRNA", "bionSBM_mRNA_lncRNA"]

if os.path.isfile(file_name)== False:
	for exp in exps:
		for run in range(0,25):
		    for level in [0,1,2,3,4,5,6]:
		        file=f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/{exp}_level_{level}_clusters.tsv.gz"
		        if os.path.isfile(file):
		            d=pd.read_csv(file, sep="\t", index_col=0)
		            data[f"{exp}_Level_{level}_Run_{run}"]=np.nan
		            data.loc[d.columns,f"{exp}_Level_{level}_Run_{run}"]=np.argmax(d.values, axis=0)
		        else:
		            print(exp, run, level, "NOT FOUND")
	data.to_csv(file_name, sep="\t", compression="gzip")
else:
	data=pd.read_csv(file_name, sep="\t", index_col=0)

df=pd.DataFrame(columns=["Exp","GT","Run","Level","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity","N_clusters", "MDL"])
df_sil=pd.DataFrame(columns=["Exp","GT","Run","Level","N_clusters", "MDL", "Space","SilClust","SilGT"])
for ct in metadata.columns: 
	for exp in exps:
	    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
	    for level in [0,1,2,3,4,5,6]:
	        for run in range(0, 25):
	            col=f"{exp}_Level_{level}_Run_{run}"
	            if col in data.columns:
	                red=data[[ct,col]].dropna()
	                if len(set(red[col].dropna())) > 1:
	                    NMI=nmi.compute_normalised_mutual_information(red[ct], red[col])
	                    NMI_geom=nmi.compute_normalised_mutual_information(red[ct], red[col], average_method="geometric")
	                    nmi_rand=0
	                    nmi_rand_geo=0
	                    for k in range(100):
	                        a=red[col].to_list()
	                        np.random.shuffle(a)
	                        nmi_rand+=nmi.compute_normalised_mutual_information(red[ct], a)/100
	                        nmi_rand_geo+=nmi.compute_normalised_mutual_information(red[ct], a, average_method="geometric")/100
	                    ari=sklearn.metrics.adjusted_rand_score(red[ct], red[col])
	                    comp=sklearn.metrics.completeness_score(red[ct], red[col])
	                    hom=sklearn.metrics.homogeneity_score(red[ct], red[col])
	                    
	                    mdl=float(pd.read_csv(f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/{exp}_entropy.txt", header=None).iat[0,0])

	                    for space in exp.split("_")[1:]:
	                        file_data=f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/{exp}_level_{level}_{space}_topics_documents.tsv.gz"
	                        top_docs=pd.read_csv(file_data, index_col=0, sep="\t")
	                        top_docs=top_docs-top_docs.mean(axis=0)
	                        sil_clu=silhouette_score(top_docs.loc[metadata.loc[top_docs.index][ct].index].fillna(0).values, 
	                                                             labels=metadata.loc[top_docs.index][ct])
	                        sil_gt=silhouette_score(top_docs.loc[red[col].index].fillna(0).values, labels=red[col])
	                        mat=[exp, ct, run, level, len(set(red[col].dropna())), mdl, space, sil_clu, sil_gt]
	                        d=pd.DataFrame(mat,index=df_sil.columns).T
	                        df_sil=pd.concat([df_sil, d])
	                        
	                    mat=[exp, ct, run, level, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom, len(set(red[col].dropna())), mdl]
	                    d=pd.DataFrame(mat,index=df.columns).T
	                    df=pd.concat([df, d])
	                else:
	                    print("		", col, f"has only {len(set(red[col].dropna()))} cluster", flush=True)
df=df.reset_index().drop("index", axis=1)
df_sil=df_sil.reset_index().drop("index", axis=1)
df.to_csv(f"Datasets/{dataset}/bionSBM/{dataset}_NMI.tsv.gz", sep="\t", compression="gzip")
df_sil.to_csv(f"Datasets/{dataset}/bionSBM/{dataset}_Silhouette.tsv.gz", sep="\t", compression="gzip")
	
'''
#+++++++++++++++++++++++++++++++ ShareTopic +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"ShareTopic", flush=True)
if dataset not in ["BMMCCite", "Spleen"]:
	metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", index_col=0, sep="\t")
	data=metadata.copy()
	exps=["ShareTopic_Peak_GEX", "ShareTopic_Peak_mRNA", "ShareTopic_Peak_lncRNA"]
	file_name=f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_clustering.tsv.gz"
	if os.path.isfile(file_name)== False:
		for exp in exps:
		    for run in range(0,25):
		        file=f"Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
		        if os.path.isfile(file):
		            d=pd.read_csv(file, sep="\t", index_col=0)
		            data[f"{exp}_Run_{run}"]=np.nan
		            data.loc[d.columns,f"{exp}_Run_{run}"]=np.argmax(d.values, axis=0)
		        else:
		            print(exp, run, "NOT FOUND")
		data.to_csv(file_name, sep="\t", compression="gzip")
	else:
		data=pd.read_csv(file_name, sep="\t", index_col=0)

	df=pd.DataFrame(columns=["Exp","GT","Run","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity", "SilClust","SilGT"])
	for ct in metadata.columns:
	    for exp in exps:
	        print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
	        for run in range(0, 25):
	            col=f"{exp}_Run_{run}"
	            if col in data.columns:
	                file_data=f"Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
	                red=data[[ct,col]].dropna()
	                if os.path.isfile(file_data):
	                    temp=pd.read_csv(file_data, index_col=0, sep="\t").T
	                    temp=temp-temp.mean(axis=0)
	                    sil_clu=silhouette_score(temp.loc[metadata.loc[temp.index][ct].index], labels=metadata.loc[temp.index][ct])
	                    sil_gt=silhouette_score(temp.loc[red[col].index], labels=red[col])
	                else:
	                    print(f"{file_data} data NOT found")
	                    sil_clu=np.nan
	                    sil_gt=np.nan
	                                    
	                NMI=nmi.compute_normalised_mutual_information(red[ct], red[col])
	                NMI_geom=nmi.compute_normalised_mutual_information(red[ct], red[col], average_method="geometric")
	                nmi_rand=0
	                nmi_rand_geo=0
	                for k in range(100):
	                    a=red[col].to_list()
	                    np.random.shuffle(a)
	                    nmi_rand+=nmi.compute_normalised_mutual_information(red[ct], a)/100
	                    nmi_rand_geo+=nmi.compute_normalised_mutual_information(red[ct], a, average_method="geometric")/100
	                ari=sklearn.metrics.adjusted_rand_score(red[ct], red[col])
	                comp=sklearn.metrics.completeness_score(red[ct], red[col])
	                hom=sklearn.metrics.homogeneity_score(red[ct], red[col])
	                d=pd.DataFrame([exp, ct, run, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom, sil_clu, sil_gt], index=df.columns).T
	                df=pd.concat([df, d])
	df=df.reset_index().drop("index", axis=1)
	df.to_csv(f"Datasets/{dataset}/ShareTopic/{dataset}_NMI_Silhouette.tsv.gz", sep="\t", compression="gzip")


#+++++++++++++++++++++++++++++++ Mowgli +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Mowgli", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", index_col=0, sep="\t")
data=metadata.copy()
if dataset in ["BMMCCite", "Spleen"]:
	exps = ["Mowgli_ADT_GEX", "Mowgli_ADT_mRNA", "Mowgli_ADT_lncRNA", "Mowgli_ADT_mRNA_lncRNA", "Mowgli_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
else:
	exps = ["Mowgli_Peak_GEX", "Mowgli_Peak_mRNA", "Mowgli_Peak_lncRNA", "Mowgli_Peak_mRNA_lncRNA", "Mowgli_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]

file_name=f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_clustering.tsv.gz"
if os.path.isfile(file_name)== False:
	for exp in exps:
		for run in range(0,25):
		    file=f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
		    if os.path.isfile(file):
		        d=pd.read_csv(file, sep="\t", index_col=0)
		        data[f"{exp}_Run_{run}"]=np.nan
		        data.loc[d.columns,f"{exp}_Run_{run}"]=np.argmax(d.values, axis=0)
		    else:
		        print(exp, run, "NOT FOUND")
	data.to_csv(file_name, sep="\t", compression="gzip")
else:
	data=pd.read_csv(file_name, sep="\t", index_col=0)

df=pd.DataFrame(columns=["Exp","GT","Run","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity", "SilClust","SilGT"])
for ct in metadata.columns:
	for exp in exps:
	    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
	    for run in range(0, 25):
	        col=f"{exp}_Run_{run}"
	        if col in data.columns:
	            file_data=f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
	            red=data[[ct,col]].dropna()
	            if os.path.isfile(file_data):
	                temp=pd.read_csv(file_data, index_col=0, sep="\t").T
	                temp=temp-temp.mean(axis=0)
	                sil_clu=silhouette_score(temp.loc[metadata.loc[temp.index][ct].index].fillna(0).values,  labels=metadata.loc[temp.index][ct])
	                sil_gt=silhouette_score(temp.loc[red[col].index].fillna(0).values, labels=red[col])
	            else:
	                print(f"{file_data} data NOT found")
	                sil_clu=np.nan
	                sil_gt=np.nan
	                                
	            NMI=nmi.compute_normalised_mutual_information(red[ct], red[col])
	            NMI_geom=nmi.compute_normalised_mutual_information(red[ct], red[col], average_method="geometric")
	            nmi_rand=0
	            nmi_rand_geo=0
	            for k in range(100):
	                a=red[col].to_list()
	                np.random.shuffle(a)
	                nmi_rand+=nmi.compute_normalised_mutual_information(red[ct], a)/100
	                nmi_rand_geo+=nmi.compute_normalised_mutual_information(red[ct], a, average_method="geometric")/100
	            ari=sklearn.metrics.adjusted_rand_score(red[ct], red[col])
	            comp=sklearn.metrics.completeness_score(red[ct], red[col])
	            hom=sklearn.metrics.homogeneity_score(red[ct], red[col])
	            d=pd.DataFrame([exp, ct, run, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom, sil_clu, sil_gt], index=df.columns).T
	            df=pd.concat([df, d])
df=df.reset_index().drop("index", axis=1)
df.to_csv(f"Datasets/{dataset}/Mowgli/{dataset}_NMI_Silhouette.tsv.gz", sep="\t", compression="gzip")
'''
