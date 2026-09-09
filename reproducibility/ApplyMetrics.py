import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import glob
import os
import random
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import anndata as ad
import scanpy as sc
import sys
import time
import nmi

from pathlib import Path


dataset=sys.argv[1]
print("Dataset:", dataset, flush=True)
metadata=pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", index_col=0, sep="\t")


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

	                    mat=[exp, ct, run, level, NMI, nmi_rand, NMI_geom, nmi_rand_geo, ari, comp, hom, len(set(red[col].dropna())), mdl]
	                    d=pd.DataFrame(mat,index=df.columns).T
	                    df=pd.concat([df, d])
	                else:
	                    print("		", col, f"has only {len(set(red[col].dropna()))} cluster", flush=True)
df=df.reset_index().drop("index", axis=1)
df.to_csv(f"Datasets/{dataset}/bionSBM/{dataset}_NMI.tsv.gz", sep="\t", compression="gzip")
	

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
	df.to_csv(f"Datasets/{dataset}/ShareTopic/{dataset}_NMI.tsv.gz", sep="\t", compression="gzip")


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
df.to_csv(f"Datasets/{dataset}/Mowgli/{dataset}_NMI.tsv.gz", sep="\t", compression="gzip")

#++++++++++++++++++++ MOFA+, MultiVI and totalVI ++++++++++++++++++++
# These methods return continuous cell embeddings. We evaluate:
#   1) Leiden with a fixed graph/resolution, independent of ground truth.
#   2) K-means with k equal to the number of labels in each annotation column.
N_SHUFFLES = 100


def score_embedding(exp, gt, run, clustering, y_true, y_pred):
    """Compute the same clustering metrics used above."""
    nmi_value = nmi.compute_normalised_mutual_information(y_true, y_pred)
    nmi_geo = nmi.compute_normalised_mutual_information(
        y_true, y_pred, average_method="geometric"
    )

    rng = np.random.default_rng(run)
    shuffled = [rng.permutation(y_pred) for _ in range(N_SHUFFLES)]
    nmi_rand = np.mean([
        nmi.compute_normalised_mutual_information(y_true, labels)
        for labels in shuffled
    ])
    nmi_rand_geo = np.mean([
        nmi.compute_normalised_mutual_information(
            y_true, labels, average_method="geometric"
        )
        for labels in shuffled
    ])

    return {
        "Exp": exp,
        "GT": gt,
        "Run": run,
        "Clustering": clustering,
        "N_labels": pd.Series(y_true).nunique(),
        "N_clusters": pd.Series(y_pred).nunique(),
        "NMI": nmi_value,
        "NMI*": nmi_rand,
        "NMIg": nmi_geo,
        "NMIg*": nmi_rand_geo,
        "ARI": sklearn.metrics.adjusted_rand_score(y_true, y_pred),
        "Completeness": sklearn.metrics.completeness_score(y_true, y_pred),
        "Homogeneity": sklearn.metrics.homogeneity_score(y_true, y_pred),
    }


def evaluate_embedding_method(method, experiments):
    """Cluster each saved cells-by-latent embedding and write one metrics table."""
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), method, flush=True)
    rows = []

    for omics in experiments:
        exp = f"{method}_{omics}"

        for run in range(25):
            file = f"Datasets/{dataset}/{method}/{exp}/Runs/Run{run}/"f"{dataset}_{exp}_Embedding.tsv.gz"
            if not os.path.isfile(file):
                print(exp, run, "NOT FOUND", flush=True)
                continue

            # Files are latent dimensions x cells; transpose to cells x dimensions.
            embedding = pd.read_csv(file, sep="\t", index_col=0).T
            embedding = embedding.loc[~embedding.index.duplicated() & embedding.index.isin(metadata.index)]
            embedding = embedding.replace([np.inf, -np.inf], np.nan).dropna()

            if len(embedding) < 3:
                print(exp, run, "TOO FEW CELLS", flush=True)
                continue

            # Standardization prevents high-variance latent dimensions dominating distances.
            x_all = StandardScaler().fit_transform(embedding)
            latent = ad.AnnData(x_all)
            sc.pp.neighbors(latent, use_rep="X", random_state=run)
            sc.tl.leiden(latent, random_state=run, key_added="cluster")
            leiden = pd.Series(latent.obs["cluster"].astype(str).to_numpy(), index=embedding.index)

            for gt in metadata.columns:
                y_true = metadata.loc[embedding.index, gt].dropna()
                n_labels = y_true.nunique()

                # Clustering metrics are not informative for one class or one class per cell.
                if not 1 < n_labels < len(y_true):
                    continue

                positions = embedding.index.get_indexer(y_true.index)
                x = x_all[positions]

                y_leiden = leiden.loc[y_true.index].to_numpy()
                rows.append(score_embedding(exp, gt, run, "Leiden", y_true.to_numpy(), y_leiden))

                y_kmeans = KMeans(n_clusters=n_labels, n_init=20,random_state=run).fit_predict(x)
                rows.append(score_embedding(exp, gt, run, "KMeans", y_true.to_numpy(), y_kmeans))

    result = pd.DataFrame(rows)
    result.to_csv(f"Datasets/{dataset}/{method}/{dataset}_NMI.tsv.gz", sep="\t", compression="gzip", index=False)


if dataset in ["BMMCCite", "Spleen"]:
    common_experiments = [
        "ADT_GEX",
        "ADT_mRNA",
        "ADT_lncRNA",
        "ADT_mRNA_lncRNA",
        "mRNA_lncRNA",
    ]
    evaluate_embedding_method("MOFA", common_experiments)
    evaluate_embedding_method("TotalVI", ["ADT_GEX", "ADT_mRNA", "ADT_lncRNA"])
else:
    common_experiments = [
        "Peak_GEX",
        "Peak_mRNA",
        "Peak_lncRNA",
        "Peak_mRNA_lncRNA",
        "mRNA_lncRNA",
    ]
    evaluate_embedding_method("MOFA", common_experiments)
    evaluate_embedding_method("MultiVI", ["Peak_GEX", "Peak_mRNA", "Peak_lncRNA"])


#+++++++++++++++++++++++++++++++ COBOLT +++++++++++++++++++++++++++++++
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "COBOLT", flush=True)

data = metadata.copy()

if dataset in ["BMMCCite", "Spleen"]:
    exps = [
        "COBOLT_ADT_GEX",
        "COBOLT_ADT_mRNA",
        "COBOLT_ADT_lncRNA",
        "COBOLT_ADT_mRNA_lncRNA",
        "COBOLT_mRNA_lncRNA",
    ]
else:
    exps = [
        "COBOLT_Peak_GEX",
        "COBOLT_Peak_mRNA",
        "COBOLT_Peak_lncRNA",
        "COBOLT_Peak_mRNA_lncRNA",
        "COBOLT_mRNA_lncRNA",
    ]

file_name = f"Datasets/{dataset}/COBOLT/{dataset}_COBOLT_clustering.tsv.gz"

if not os.path.isfile(file_name):
    for exp in exps:
        for run in range(25):
            file = f"Datasets/{dataset}/COBOLT/{exp}/Runs/Run{run}/{dataset}_{exp}_Embedding.tsv.gz"

            if os.path.isfile(file):
                embedding = pd.read_csv(file, sep="\t", index_col=0)

                # COBOLT files are dimensions x cells. Assign each cell to
                # the latent category with the largest coordinate.
                col = f"{exp}_Run_{run}"
                data[col] = np.nan
                data.loc[embedding.columns, col] = np.argmax(embedding.values, axis=0)
            else:
                print(exp, run, "NOT FOUND", flush=True)

    data.to_csv(file_name, sep="\t", compression="gzip")
else:
    data = pd.read_csv(file_name, sep="\t", index_col=0)


columns = [
    "Exp", "GT", "Run",
    "NMI", "NMI*", "NMIg", "NMIg*",
    "ARI", "Completeness", "Homogeneity",
]
rows = []

for ct in metadata.columns:
    for exp in exps:
        print(
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            "NMI",
            ct,
            exp,
            flush=True,
        )

        for run in range(25):
            col = f"{exp}_Run_{run}"

            if col not in data.columns:
                continue

            red = data[[ct, col]].dropna()

            if red[col].nunique() < 2:
                print(
                    col,
                    f"has only {red[col].nunique()} cluster",
                    flush=True,
                )
                continue


            nmi_value = nmi.compute_normalised_mutual_information(
                red[ct],
                red[col],
            )
            nmi_geom = nmi.compute_normalised_mutual_information(
                red[ct],
                red[col],
                average_method="geometric",
            )

            nmi_rand = 0
            nmi_rand_geom = 0

            for _ in range(100):
                shuffled = red[col].to_numpy().copy()
                np.random.shuffle(shuffled)

                nmi_rand += (
                    nmi.compute_normalised_mutual_information(
                        red[ct],
                        shuffled,
                    )
                    / 100
                )
                nmi_rand_geom += (
                    nmi.compute_normalised_mutual_information(
                        red[ct],
                        shuffled,
                        average_method="geometric",
                    )
                    / 100
                )

            rows.append([
                exp,
                ct,
                run,
                nmi_value,
                nmi_rand,
                nmi_geom,
                nmi_rand_geom,
                sklearn.metrics.adjusted_rand_score(red[ct], red[col]),
                sklearn.metrics.completeness_score(red[ct], red[col]),
                sklearn.metrics.homogeneity_score(red[ct], red[col]),
            ])

df = pd.DataFrame(rows, columns=columns)
df.to_csv(
    f"Datasets/{dataset}/COBOLT/{dataset}_NMI.tsv.gz",
    sep="\t",
    compression="gzip",
    index=False,
)
