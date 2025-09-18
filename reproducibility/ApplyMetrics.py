import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import scanpy as sc
import numpy as np

import glob
import os
import random
import sklearn
import sys
import time

from helps import *
from pathlib import Path

import nmi

dataset=sys.argv[1]
print("Dataset:", dataset, flush=True)

#+++++++++++++++++++++++++++++++ SBM +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"hSBM and nSBM", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
df=metadata.copy()

if dataset in ["BMMCCite", "Spleen"]:
	exps_hSBM = ["hSBM_ADT","hSBM_GEX","hSBM_mRNA", "hSBM_lncRNA"]
	exps_nSBM = ["nSBM_ADT_GEX", "nSBM_ADT_mRNA", "nSBM_ADT_lncRNA", "nSBM_ADT_mRNA_lncRNA", "nSBM_mRNA_lncRNA"]

else:
	exps_hSBM = ["hSBM_Peak","hSBM_GEX","hSBM_mRNA", "hSBM_lncRNA"]
	exps_nSBM = ["nSBM_Peak_GEX", "nSBM_Peak_mRNA", "nSBM_Peak_lncRNA", "nSBM_Peak_mRNA_lncRNA", "nSBM_mRNA_lncRNA"]

file_name=f"Datasets/{dataset}/SBM/{dataset}_SBMs_25Run.tsv.gz"

if os.path.isfile(file_name) == False:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "hSBM", flush=True)
	for exp in exps_hSBM:
		print(exp, flush=True)
		for run in range(0, 25):
			for level in range(0, 6):
				file_data=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_cluster_level_{level}.txt"
				if os.path.isfile(file_data):
					with open(file_data) as f:
						diz = json.load(f)
					df[f"{exp}_Level_{level}_Run_{run}"]=np.nan
					for key in diz.keys():
						df[f"{exp}_Level_{level}_Run_{run}"].loc[np.array(diz[key])[:, 0]]=str(key)
				else:
					print(f"File {file_data} not found", flush=True)

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "nSBM", flush=True)
	for exp in exps_nSBM:
		print(exp, flush=True)
		for run in range(0, 25):
			for level in range(0, 6): 
				file_data=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{level}_clusters.csv.gz"
				if os.path.isfile(file_data):
					temp=pd.read_csv(file_data, index_col=0)
					df[f"{exp}_Level_{level}_Run_{run}"]=np.nan
					df[f"{exp}_Level_{level}_Run_{run}"].loc[temp.columns]=np.argmax(temp.values, axis=0).astype(str)
				else:
					print(f"File {file_data} not found", flush=True)

	df.to_csv(file_name, compression="gzip", sep="\t")

data=pd.read_csv(file_name, sep="\t", index_col=0)

exps = exps_hSBM + exps_nSBM

file_name=f"Datasets/{dataset}/SBM/{dataset}_SBMs_25Run_NMI.tsv.gz"
if os.path.isfile(file_name) == False:
	df=pd.DataFrame(columns=["Exp","GT","Run","Level","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity","N_clusters", "MDL", "Space","SilClust","SilGT"])
	for ct in metadata.columns: 
		for exp in exps:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
			for level in range(0,6):
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
							
							mdl=float(pd.read_csv(f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_entropy.txt", header=None).iat[0,0])

							if "nSBM" in exp:
								for space in exp.split("_")[1:]:
									file_data=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{level}_{space}_topics_documents.csv.gz"
									if os.path.isfile(file_data):
										top_docs=pd.read_csv(file_data, index_col=0)
										top_docs=top_docs-top_docs.mean(axis=0)
										sil_clu=sklearn.metrics.silhouette_score(top_docs.loc[metadata.loc[top_docs.index][ct].index].fillna(0).values, 
																			 labels=metadata.loc[top_docs.index][ct])
										sil_gt=sklearn.metrics.silhouette_score(top_docs.loc[red[col].index].fillna(0).values, labels=red[col])
									else:
										print(f"{file_data} data NOT found")
										sil_clu=np.nan
										sil_gt=np.nan
							elif "hSBM" in exp:
								space=exp.split("_")[1]
								file_data=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_topsbm_level_{level}_topic_dist.csv.gz"
								if os.path.isfile(file_data):
									top_docs=pd.read_csv(file_data, index_col=0)
									top_docs=top_docs-top_docs.mean(axis=0)
									sil_clu=sklearn.metrics.silhouette_score(top_docs.loc[metadata.loc[top_docs.index][ct].index].fillna(0).values, 
																			 labels=metadata.loc[top_docs.index][ct])
									sil_gt=sklearn.metrics.silhouette_score(top_docs.loc[red[col].index].fillna(0).values, labels=red[col])
								else:
									print(f"{file_data} data NOT found")
									sil_clu=np.nan
									sil_gt=np.nan
									
							mat=[exp, ct, run, level, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom, 
								 len(set(red[col].dropna())), mdl, space, sil_clu, sil_gt]
							d=pd.DataFrame(mat,index=df.columns).T
							df=pd.concat([df, d])
						else:
							print("		", col, f"has only {len(set(red[col].dropna()))} cluster", flush=True)

	df=df.reset_index().drop("index", axis=1)
	df["NMI"]=df["NMI"].astype(float)
	df["NMI*"]=df["NMI*"].astype(float)
	df["NMIg"]=df["NMIg"].astype(float)
	df["NMIg*"]=df["NMIg*"].astype(float)
	df["ARI"]=df["ARI"].astype(float)
	df["Completeness"]=df["Completeness"].astype(float)
	df["Homogeneity"]=df["Homogeneity"].astype(float)
	df["NMI/NMI*"]=df["NMI"]/df["NMI*"]
	df["NMIg/NMIg*"]=df["NMIg"]/df["NMIg*"]
	df.to_csv(file_name, sep="\t", compression="gzip")


#+++++++++++++++++++++++++++++++ ShareTopic +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "ShareTopic", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
df=metadata.copy()

exps=["ShareTopic_Peak_GEX", "ShareTopic_Peak_mRNA", "ShareTopic_Peak_lncRNA"]

if dataset not in ["BMMCCite", "Spleen"]:
	file_name=f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_25Run.tsv.gz"
	if os.path.isfile(file_name) == False:
		for exp in exps:
			print(exp, flush=True)
			for run in range(0, 25):
				file_data=f"Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
				if os.path.isfile(file_data):
					temp=pd.read_csv(file_data, index_col=0, sep="\t")
					df[f"{exp}_Run_{run}"]=np.nan
					df[f"{exp}_Run_{run}"].loc[temp.columns]=np.argmax(temp, axis=0).astype(str)
				else:
					print(f"File {file_data} not found", flush=True)

		df.to_csv(file_name, compression="gzip", sep="\t")
	
	data=pd.read_csv(file_name, sep="\t", index_col=0)

	file_name=f"Datasets/{dataset}/ShareTopic/{dataset}_ShareTopic_25Run_NMI.tsv.gz"
	if os.path.isfile(file_name) == False:
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
							sil_clu=sklearn.metrics.silhouette_score(temp.loc[metadata.loc[temp.index][ct].index].fillna(0).values, 
																	 labels=metadata.loc[temp.index][ct])
							sil_gt=sklearn.metrics.silhouette_score(temp.loc[red[col].index].fillna(0).values, labels=red[col])
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
		df["NMI"]=df["NMI"].astype(float)
		df["NMI*"]=df["NMI*"].astype(float)
		df["NMIg"]=df["NMIg"].astype(float)
		df["NMIg*"]=df["NMIg*"].astype(float)
		df["ARI"]=df["ARI"].astype(float)
		df["Completeness"]=df["Completeness"].astype(float)
		df["Homogeneity"]=df["Homogeneity"].astype(float)
		df["NMI/NMI*"]=df["NMI"]/df["NMI*"]
		df["NMIg/NMIg*"]=df["NMIg"]/df["NMIg*"]
		df.to_csv(file_name, sep="\t", compression="gzip")

'''
#+++++++++++++++++++++++++++++++ cisTopic +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "cisTopic", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
df=metadata.copy()


if dataset == "BMMCCite":
	exps = ["cisTopic_GEX","cisTopic_ADT"]
else:
	exps = ["cisTopic_Peak","cisTopic_GEX","cisTopic_mRNA", "cisTopic_lncRNA"]


file_name=f"Datasets/{dataset}/cisTopic/{dataset}_cisTopic_25Run.tsv.gz"
if os.path.isfile(file_name) == False:
	for exp in exps:
		print(exp, flush=True)
		for run in range(0, 25):
			file=f"Datasets/{dataset}/cisTopic/{exp}/Runs/Run{run}/{dataset}_{exp}_Topic_Cell.tsv.gz"
			if os.path.isfile(file):
				temp=pd.read_csv(file, index_col=0, sep="\t")
				df[f"{exp}_Run_{run}"]=np.nan
				df[f"{exp}_Run_{run}"].loc[temp.columns]=np.argmax(temp, axis=0).astype(str)

	df.to_csv(file_name, compression="gzip", sep="\t")

data=pd.read_csv(file_name, sep="\t", index_col=0)

file_name=f"Datasets/{dataset}/cisTopic/{dataset}_cisTopic_25Run_NMI.tsv.gz"
if os.path.isfile(file_name) == False:
	df=pd.DataFrame(columns=["Exp","GT","Run","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity"])
	for ct in metadata.columns:
		for exp in exps:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
			for run in range(0, 25):
				col=f"{exp}_Run_{run}"
				if col in data.columns:
					red=data[[ct,col]].dropna()
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
					d=pd.DataFrame([exp, ct, run, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom], index=df.columns).T
					df=pd.concat([df, d])

	df=df.reset_index().drop("index", axis=1)
	df["NMI"]=df["NMI"].astype(float)
	df["NMI*"]=df["NMI*"].astype(float)
	df["NMIg"]=df["NMIg"].astype(float)
	df["NMIg*"]=df["NMIg*"].astype(float)
	df["ARI"]=df["ARI"].astype(float)
	df["Completeness"]=df["Completeness"].astype(float)
	df["Homogeneity"]=df["Homogeneity"].astype(float)
	df["NMI/NMI*"]=df["NMI"]/df["NMI*"]
	df["NMIg/NMIg*"]=df["NMIg"]/df["NMIg*"]
	df.to_csv(file_name, sep="\t", compression="gzip")
'''
#+++++++++++++++++++++++++++++++ Mowgli +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Mowgli", flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
df=metadata.copy()

if dataset in ["BMMCCite", "Spleen"]:
	exps = ["Mowgli_ADT_GEX", "Mowgli_ADT_mRNA", "Mowgli_ADT_lncRNA", "Mowgli_ADT_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]
else:
	exps = ["Mowgli_Peak_GEX", "Mowgli_Peak_mRNA", "Mowgli_Peak_lncRNA", "Mowgli_Peak_mRNA_lncRNA", "Mowgli_mRNA_lncRNA"]

file_name=f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_25Run.tsv.gz"
if os.path.isfile(file_name) == False:
	for exp in exps:
		print(exp, flush=True)
		for run in range(0, 25):
			file_data=f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
			if os.path.isfile(file_data):
				temp=pd.read_csv(file_data, index_col=0, sep="\t")
				df[f"{exp}_Run_{run}"]=np.nan
				df[f"{exp}_Run_{run}"].loc[temp.columns]=np.argmax(temp, axis=0).astype(str)
			else:
				print(f"File {file_data} not found", flush=True)
	df.to_csv(file_name, sep="\t", compression="gzip")

data=pd.read_csv(file_name, sep="\t", index_col=0)

file_name=f"Datasets/{dataset}/Mowgli/{dataset}_Mowgli_25Run_NMI.tsv.gz"
if os.path.isfile(file_name) == False:
	df=pd.DataFrame(columns=["Exp","GT","Run","NMI","NMI*","NMIg","NMIg*", "ARI","Completeness","Homogeneity","SilClust","SilGT"])
	for ct in metadata.columns:
		for exp in exps:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "NMI", ct, exp, flush=True)
			for run in range(0, 25):
				col=f"{exp}_Run_{run}"
				if col in data.columns:
					red=data[[ct,col]].dropna()
					file_data=f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"
					if os.path.isfile(file_data):
						temp=pd.read_csv(file_data, index_col=0, sep="\t").T
						temp=temp-temp.mean(axis=0)
						sil_clu=sklearn.metrics.silhouette_score(temp.loc[metadata.loc[temp.index][ct].index].fillna(0).values, 
																 labels=metadata.loc[temp.index][ct])
						sil_gt=sklearn.metrics.silhouette_score(temp.loc[red[col].index].fillna(0).values, labels=red[col])
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
	df["NMI"]=df["NMI"].astype(float)
	df["NMI*"]=df["NMI*"].astype(float)
	df["NMIg"]=df["NMIg"].astype(float)
	df["NMIg*"]=df["NMIg*"].astype(float)
	df["ARI"]=df["ARI"].astype(float)
	df["Completeness"]=df["Completeness"].astype(float)
	df["Homogeneity"]=df["Homogeneity"].astype(float)
	df["NMI/NMI*"]=df["NMI"]/df["NMI*"]
	df["NMIg/NMIg*"]=df["NMIg"]/df["NMIg*"]
	df.to_csv(file_name, sep="\t", compression="gzip")
