import warnings
warnings.filterwarnings('ignore')
import os
import sys

import numpy as np
import pandas as pd
from numba import njit, prange

from helps import bh_fdr, find_best_topic_level

DATASET = sys.argv[1]
N_RUNS = 25
N_NULL = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MAX_CELLS = 3000
SEED = 0

EXPERIMENTS = {
#    "Peak_mRNA_lncRNA": "mRNA",
    "Peak_mRNA": "mRNA",
    "Peak_lncRNA": "lncRNA",
    "Peak_GEX": "GEX",
}


@njit
def center_distance_matrix(v):
    """Return the double-centred pairwise absolute-distance matrix."""
    n = len(v)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = abs(v[i] - v[j])

    row_mean = np.zeros(n)
    col_mean = np.zeros(n)
    for i in range(n):
        row_mean[i] = D[i].mean()
    for j in range(n):
        col_mean[j] = D[:, j].mean()

    total_mean = D.mean()
    for i in range(n):
        for j in range(n):
            D[i, j] -= row_mean[i] + col_mean[j] - total_mean
    return D


@njit(parallel=True)
def distcorr(X, Y):
    """Compute distance correlation for every X-topic/Y-topic pair."""
    _, nx = X.shape
    _, ny = Y.shape
    result = np.zeros((nx, ny))

    A = [center_distance_matrix(X[:, i]) for i in range(nx)]
    B = [center_distance_matrix(Y[:, j]) for j in range(ny)]
    var_x = np.array([np.mean(a * a) for a in A])
    var_y = np.array([np.mean(b * b) for b in B])

    for i in prange(nx):
        if var_x[i] == 0:
            continue
        for j in range(ny):
            if var_y[j] > 0:
                result[i, j] = np.mean(A[i] * B[j]) / np.sqrt(var_x[i] * var_y[j])
    return result


def topic_files(base_dir, exp, rna_space, level):
    """Return RNA and ATAC topic-document paths for one hierarchy level."""
    prefix = f"{base_dir}/bionSBM_{exp}_level_{level}"
    return (
        f"{prefix}_{rna_space}_topics_documents.tsv.gz",
        f"{prefix}_Peak_topics_documents.tsv.gz",
    )


def load_topic_matrices(base_dir, exp, rna_space, level):
    """Load topic matrices, using the next finer level when either has <=2 topics."""
    level = int(level)
    paths = topic_files(base_dir, exp, rna_space, level)
    if not all(os.path.isfile(path) for path in paths):
        return None

    X = pd.read_csv(paths[0], sep="\t", index_col=0)
    Y = pd.read_csv(paths[1], sep="\t", index_col=0)

    if (X.shape[1] <= 2 or Y.shape[1] <= 2) and level > 0:
        fallback = topic_files(base_dir, exp, rna_space, level - 1)
        if all(os.path.isfile(path) for path in fallback):
            level -= 1
            X = pd.read_csv(fallback[0], sep="\t", index_col=0)
            Y = pd.read_csv(fallback[1], sep="\t", index_col=0)

    # Remove constant/invalid topics before computing distances.
    X = X.loc[:, X.var() > 1e-8].dropna(axis=1, how="all").dropna(axis=0)
    Y = Y.loc[:, Y.var() > 1e-8].dropna(axis=1, how="all").dropna(axis=0)
    return level, X, Y


meta = pd.read_csv(
    f"Datasets/{DATASET}/{DATASET}_Metadata.tsv.gz", sep="\t", index_col=0
)
n_celltypes = meta["CellType"].nunique()

for exp_index, (exp, rna_space) in enumerate(EXPERIMENTS.items()):
    rows = []

    for run in range(N_RUNS):
        print(f"{DATASET} | {exp} | Run {run}", flush=True)
        base_dir = f"Datasets/{DATASET}/bionSBM/bionSBM_{exp}/Runs/Run{run}"
        if not os.path.isdir(base_dir):
            continue

        level = find_best_topic_level(base_dir, n_celltypes)
        loaded = None if level is None else load_topic_matrices(
            base_dir, exp, rna_space, level
        )
        if loaded is None:
            continue
        level, X, Y = loaded

        cells = X.index.intersection(Y.index)
        if not len(cells) or not X.shape[1] or not Y.shape[1]:
            continue

        # Reproducible 3,000-cell maximum, as described in the Methods.
        rng = np.random.default_rng(np.random.SeedSequence([SEED, exp_index, run]))
        if len(cells) > MAX_CELLS:
            cells = pd.Index(rng.choice(cells.to_numpy(), MAX_CELLS, replace=False))

        X = X.loc[cells] - X.loc[cells].mean(axis=0)
        Y = Y.loc[cells] - Y.loc[cells].mean(axis=0)
        Xv, Yv = X.to_numpy(), Y.to_numpy()

        real = distcorr(Xv, Yv)
        null = np.empty((N_NULL, *real.shape))
        for permutation in range(N_NULL):
            print(f"  Null {permutation + 1}/{N_NULL}", flush=True)
            # One shared row permutation preserves all within-ATAC relationships.
            null[permutation] = distcorr(Xv, Yv[rng.permutation(len(Yv))])

        null_mean = null.mean(axis=0)
        null_std = null.std(axis=0)

        for rna_topic in range(real.shape[0]):
            for atac_topic in range(real.shape[1]):
                observed = real[rna_topic, atac_topic]
                values = null[:, rna_topic, atac_topic]
                mu, sigma = null_mean[rna_topic, atac_topic], null_std[rna_topic, atac_topic]
                rows.append({
                    "dataset": DATASET,
                    "exp": exp,
                    "run": run,
                    "level": level,
                    "RNA_space": rna_space,
                    "RNA_topic": rna_topic,
                    "ATAC_topic": atac_topic,
                    "n_cells": len(cells),
                    "dcor_real": observed,
                    "dcor_null_mean": mu,
                    "dcor_null_std": sigma,
                    "delta_dcor": observed - mu,
                    "zscore": (observed - mu) / sigma if sigma > 0 else np.nan,
                    "pval": (np.sum(values >= observed) + 1) / (N_NULL + 1),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["qval"] = df.groupby("run")["pval"].transform(
            lambda x: bh_fdr(x.to_numpy())
        )
        df["n_tests"] = df.groupby("run")["pval"].transform("size")

    output = f"Datasets/{DATASET}/bionSBM/{exp}/{exp}_dCorr_ATAC_GEX_results.tsv"
    df.to_csv(output, sep="\t", index=False)
    print(f"Saved: {output}", flush=True)
