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
from sklearn.metrics import adjusted_mutual_info_score as AMI

dataset=sys.argv[1]
print("Dataset -->", dataset, flush=True)

#Peak ++++++++++++++++++++++++++++++++
if dataset in ["PBMC","MouseSkin","BMMCMultiOme","HSPC"]:
	peak=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_Peak_Def.h5ad").var
	exps=["hSBM_Peak","nSBM_Peak_GEX", "nSBM_Peak_mRNA", "nSBM_Peak_lncRNA", "nSBM_Peak_mRNA_lncRNA"]
	data=pd.DataFrame(columns=["Dataset","Exp","Level","Topic","Number of peaks","Run"])
	filename=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_Peak.tsv.gz"
	if os.path.isfile(filename)==False:
		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
			for l in range(0,4):
				file=f'Datasets/{dataset}/SBM/hSBM_Peak/Runs/Run{run}/{dataset}_hSBM_Peak_topics_level_{l}.txt'
				if os.path.isfile(file):
					with open(file) as f:
						d = json.load(f)
					d1={key : flat_list(np.array(d[key])[:,:1].tolist()).tolist() for key in d.keys()}
					temp=pd.DataFrame.from_dict(d1, orient="index").T
					for col in temp.columns:
						to_append=pd.DataFrame(index=data.columns, data=[dataset, "hSBM_Peak", l, f"Topic_{col}", len(temp[col].dropna()), run]).T
						data=pd.concat([data, to_append])

					peak[f"hSBM_Peak_Run_{run}_Level_{l}"]=np.nan
					for col in temp.columns:
						for p in list(temp[col].dropna()):
							peak.at[p, f"hSBM_Peak_Run_{run}_Level_{l}"]=str(col)

		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
			for exp in exps[1:]:
				for l in range(0, 4):
					file=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{l}_Peak_topics.csv.gz"
					if os.path.isfile(file):
						temp=pd.read_csv(file, index_col=0)
						for col in temp.columns:
							to_append=pd.DataFrame(index=data.columns, data=[dataset, exp, l, f"Topic_{list(temp.columns).index(col)}", len(temp[col][temp[col]!=0]), run]).T
							data=pd.concat([data, to_append])

						if temp.index[0][:3]!="chr":
							print("Problem", flush=True)
						inter=intersection([peak.index, temp.index])
						temp=temp.loc[inter]
						peak[f"{exp}_Run_{run}_Level_{l}"]=np.nan
						peak[f"{exp}_Run_{run}_Level_{l}"].loc[inter]=np.argmax(temp.values, axis=1).astype(str)

		peak.to_csv(filename, compression="gzip", sep="\t")
		data.reset_index().drop("index", axis=1).to_csv(f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_Peak_Data.tsv.gz", compression="gzip", sep="\t")

	filename1=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_Peak_NMI.tsv.gz"
	if os.path.isfile(filename1)==False:
		data=pd.read_csv(filename, sep="\t", index_col=0)
		df=pd.DataFrame(index=exps, columns=exps)
		df=df.fillna(0)
		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), run, flush=True)
			t=pd.DataFrame(index=exps, columns=exps)
			for exp in t.index:
				for ex in t.columns[list(t.index).index(exp)+1:]:
					count=0
					ami=0
					for l in range(0,4):
						a=f"{exp}_Run_{run}_Level_{l}"
						b=f"{ex}_Run_{run}_Level_{l}"
						if a in data.columns and b in data.columns:
							ami+=AMI(data[a].astype(str), data[b].astype(str))
							count+=1
						else:
							print(a, "or", b, "not found", flush=True)
					if count != 0:
						t.at[exp, ex]=ami/count
					else:
						t.at[exp, ex]=0
			df=df+t
		df=df.astype(float)
		df.to_csv(filename1, compression="gzip", sep="\t")

#ADT +++++++++++++++++++++++++++++++++
elif dataset in ["BMMCCite","Spleen"]:
	print("I am under the elif", flush=True)
	adt=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_ADT_Def.h5ad").var
	exps=["hSBM_ADT","nSBM_ADT_GEX", "nSBM_ADT_mRNA", "nSBM_ADT_lncRNA", "nSBM_ADT_mRNA_lncRNA"]
	data=pd.DataFrame(columns=["Dataset","Exp","Level","Topic","Number of ADTs","Run"])
	filename=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_ADT.tsv.gz"
	if os.path.isfile(filename)==False:
		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
			for l in range(0,4):
				file=f'Datasets/{dataset}/SBM/hSBM_ADT/Runs/Run{run}/{dataset}_hSBM_ADT_topics_level_{l}.txt'
				if os.path.isfile(file):
					with open(file) as f:
						d = json.load(f)
					d1={key : flat_list(np.array(d[key])[:,:1].tolist()).tolist() for key in d.keys()}
					temp=pd.DataFrame.from_dict(d1, orient="index").T
					for col in temp.columns:
						to_append=pd.DataFrame(index=data.columns, data=[dataset, "hSBM_ADT", l, f"Topic_{col}", len(temp[col].dropna()), run]).T
						data=pd.concat([data, to_append])

					adt[f"hSBM_ADT_Run_{run}_Level_{l}"]=np.nan
					for col in temp.columns:
						for p in list(temp[col].dropna()):
							adt.at[p, f"hSBM_ADT_Run_{run}_Level_{l}"]=str(col)

		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
			for exp in exps[1:]:
				for l in range(0, 4):
					file=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{l}_ADT_topics.csv.gz"
					if os.path.isfile(file):
						temp=pd.read_csv(file, index_col=0)
						for col in temp.columns:
							to_append=pd.DataFrame(index=data.columns, data=[dataset, exp, l, f"Topic_{list(temp.columns).index(col)}", len(temp[col][temp[col]!=0]), run]).T
							data=pd.concat([data, to_append])

						inter=intersection([adt.index, temp.index])
						temp=temp.loc[inter]
						adt[f"{exp}_Run_{run}_Level_{l}"]=np.nan
						adt[f"{exp}_Run_{run}_Level_{l}"].loc[inter]=np.argmax(temp.values, axis=1).astype(str)

		adt.to_csv(filename, compression="gzip", sep="\t")
		data.reset_index().drop("index", axis=1).to_csv(f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_ADT_Data.tsv.gz", compression="gzip", sep="\t")

	filename1=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_ADT_NMI.tsv.gz"
	if os.path.isfile(filename1)==False:
		data=pd.read_csv(filename, sep="\t", index_col=0)
		df=pd.DataFrame(index=exps, columns=exps)
		df=df.fillna(0)
		for run in range(0, 25):
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), run, flush=True)
			t=pd.DataFrame(index=exps, columns=exps)
			for exp in t.index:
				for ex in t.columns[list(t.index).index(exp)+1:]:
					count=0
					ami=0
					for l in range(0,4):
						a=f"{exp}_Run_{run}_Level_{l}"
						b=f"{ex}_Run_{run}_Level_{l}"
						if a in data.columns and b in data.columns:
							ami+=AMI(data[a].astype(str), data[b].astype(str))
							count+=1
						else:
							print(a, "or", b, "not found", flush=True)
					if count != 0:
						t.at[exp, ex]=ami/count
					else:
						t.at[exp, ex]=0
			df=df+t
		df=df.astype(float)
		df.to_csv(filename1, compression="gzip", sep="\t")


#lncRNA +++++++++++++++++++++++++++++++++
lncrna=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_lncRNA_Def.h5ad").var
if dataset in ["BMMCCite", "Spleen"]:
	exps=["hSBM_lncRNA", "nSBM_ADT_lncRNA", "nSBM_mRNA_lncRNA", "nSBM_ADT_mRNA_lncRNA"]
else:
	exps=["hSBM_lncRNA", "nSBM_Peak_lncRNA", "nSBM_mRNA_lncRNA", "nSBM_Peak_mRNA_lncRNA"]

filename=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_lncRNA.tsv.gz"
if os.path.isfile(filename)==False:
	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
		for l in range(0,4):
			file=f'Datasets/{dataset}/SBM/hSBM_lncRNA/Runs/Run{run}/{dataset}_hSBM_lncRNA_topics_level_{l}.txt'
			if os.path.isfile(file):
				with open(file) as f:
					d = json.load(f)
				d1={key : flat_list(np.array(d[key])[:,:1].tolist()).tolist() for key in d.keys()}
				temp=pd.DataFrame.from_dict(d1, orient="index").T
				if temp.shape[1] > 1:
					lncrna[f"hSBM_lncRNA_Run_{run}_Level_{l}"]=np.nan
					for col in temp.columns:
						for p in list(temp[col].dropna()):
							lncrna.at[p, f"hSBM_lncRNA_Run_{run}_Level_{l}"]=str(col)

	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
		for exp in exps[1:]:
			for l in range(0, 4):
				file=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{l}_lncRNA_topics.csv.gz"
				if os.path.isfile(file):
					temp=pd.read_csv(file, index_col=0)
					if temp.shape[1] > 1:
						temp.index=temp.index.str.replace("#","")
						inter=intersection([lncrna.index, temp.index])
						temp=temp.loc[inter]
						lncrna[f"{exp}_Run_{run}_Level_{l}"]=np.nan
						lncrna[f"{exp}_Run_{run}_Level_{l}"].loc[inter]=np.argmax(temp.values, axis=1).astype(str)

	lncrna.to_csv(filename, compression="gzip", sep="\t")

filename1=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_lncRNA_NMI.tsv.gz"
if os.path.isfile(filename1)==False:
	data=pd.read_csv(filename, sep="\t", index_col=0)
	df=pd.DataFrame(index=exps, columns=exps)
	df=df.fillna(0)
	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), run, flush=True)
		t=pd.DataFrame(index=exps, columns=exps)
		for exp in t.index:
			for ex in t.columns[list(t.index).index(exp)+1:]:
				count=0
				ami=0
				for l in range(0,4):
					a=f"{exp}_Run_{run}_Level_{l}"
					b=f"{ex}_Run_{run}_Level_{l}"
					if a in data.columns and b in data.columns:
						ami+=AMI(data[a].astype(str), data[b].astype(str))
						count+=1
					else:
						print(a, "or", b, "not found", flush=True)
				if count != 0:
					t.at[exp, ex]=ami/count
				else:
					t.at[exp, ex]=0
		df=df+t
	df=df.astype(float)
	df.to_csv(filename1, compression="gzip", sep="\t")


#mRNA +++++++++++++++++++++++++++++++++
mrna=sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_mRNA_Def.h5ad").var
if dataset in ["BMMCCite", "Spleen"]:
	exps=["hSBM_mRNA", "nSBM_ADT_mRNA", "nSBM_mRNA_lncRNA", "nSBM_ADT_mRNA_lncRNA"]
else:
	exps=["hSBM_mRNA", "nSBM_Peak_mRNA", "nSBM_mRNA_lncRNA", "nSBM_Peak_mRNA_lncRNA"]

filename=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_mRNA.tsv.gz"
if os.path.isfile(filename)==False:
	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
		for l in range(0,4):
			file=f'Datasets/{dataset}/SBM/hSBM_mRNA/Runs/Run{run}/{dataset}_hSBM_mRNA_topics_level_{l}.txt'
			if os.path.isfile(file):
				with open(file) as f:
					d = json.load(f)
				d1={key : flat_list(np.array(d[key])[:,:1].tolist()).tolist() for key in d.keys()}
				temp=pd.DataFrame.from_dict(d1, orient="index").T
				if temp.shape[1] > 1:
					mrna[f"hSBM_mRNA_Run_{run}_Level_{l}"]=np.nan
					for col in temp.columns:
						for p in list(temp[col].dropna()):
							mrna.at[p, f"hSBM_mRNA_Run_{run}_Level_{l}"]=str(col)

	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Run -->", run, flush=True)
		for exp in exps[1:]:
			for l in range(0, 4):
				file=f"Datasets/{dataset}/SBM/{exp}/Runs/Run{run}/{dataset}_{exp}_level_{l}_mRNA_topics.csv.gz"
				if os.path.isfile(file):
					temp=pd.read_csv(file, index_col=0)
					if temp.shape[1] > 1:
						temp.index=temp.index.str.replace("#","")
						inter=intersection([mrna.index, temp.index])
						mrna[f"{exp}_Run_{run}_Level_{l}"]=np.nan
						mrna[f"{exp}_Run_{run}_Level_{l}"].loc[inter]=np.argmax(temp.values, axis=1).astype(str)
	
	mrna.to_csv(filename, compression="gzip", sep="\t")

filename1=f"Datasets/{dataset}/SBM/25Runs_{dataset}_SBMs_Topics_mRNA_NMI.tsv.gz"
if os.path.isfile(filename1)==False:
	data=pd.read_csv(filename, sep="\t", index_col=0)

	df=pd.DataFrame(index=exps, columns=exps)
	df=df.fillna(0)
	for run in range(0, 25):
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), run, flush=True)
		t=pd.DataFrame(index=exps, columns=exps)
		for exp in t.index:
			for ex in t.columns[list(t.index).index(exp)+1:]:
				count=0
				ami=0
				for l in range(0,4):
					a=f"{exp}_Run_{run}_Level_{l}"
					b=f"{ex}_Run_{run}_Level_{l}"
					if a in data.columns and b in data.columns:
						ami+=AMI(data[a].astype(str), data[b].astype(str))
						count+=1
					else:
						print(a, "or", b, "not found", flush=True)
				if count != 0:
					t.at[exp, ex]=ami/count
				else:
					t.at[exp, ex]=0
		df=df+t
	df=df.astype(float)
	df.to_csv(filename1, compression="gzip", sep="\t")

