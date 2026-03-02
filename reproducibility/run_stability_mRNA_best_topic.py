import os, sys, time
import numpy as np
import pandas as pd
import scanpy as sc
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# USER INPUT
# ============================================================

dataset = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # top-k topics per cell type
n_runs = 25

main_fs = "Peak" if dataset not in ["BMMCCite", "Spleen"] else "ADT"
print(f"Dataset: {dataset} | Main FS: {main_fs} | top-k = {k}", flush=True)

# experiments
order = ["mRNA", "mRNA_lncRNA", f"{main_fs}_mRNA", f"{main_fs}_mRNA_lncRNA"]

# ============================================================
# Utilities
# ============================================================

def pdf_similarity(p, q):
    p = np.nan_to_num(p, nan=0.0)
    q = np.nan_to_num(q, nan=0.0)

    if p.sum() > 0: p = p / p.sum()
    if q.sum() > 0: q = q / q.sum()

    return cosine_similarity(p[None], q[None])[0, 0]
 
# ============================================================
# Load metadata
# ============================================================

meta = pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",sep="\t",index_col=0)
celltypes = meta["CellType"].astype(str)
n_celltypes = celltypes.nunique()

# feature universe (for alignment only)
features = sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_mRNA_Def.h5ad").var.index

# ============================================================
# Load runs (Z + topic-docs) with adaptive level selection
# ============================================================

Z_runs = {fs: [] for fs in order}
TD_runs = {fs: [] for fs in order}

for run in range(n_runs):
    print(time.strftime("%H:%M:%S"), "Run", run, flush=True)

    for fs in order:
        base_dir = f"Datasets/{dataset}/bionSBM/bionSBM_{fs}/Runs/Run{run}"
        if not os.path.isdir(base_dir):
            continue

        best_file = None
        best_delta = np.inf

        for fname in os.listdir(base_dir):
            if not fname.endswith(f"_mRNA_topics.tsv.gz"):
                continue
            if "_level_" not in fname:
                continue

            fpath = os.path.join(base_dir, fname)

            try:
                tmp = pd.read_csv(fpath, sep="\t", index_col=0, nrows=5)
            except Exception:
                continue

            delta = abs(tmp.shape[1] - n_celltypes)
            if delta < best_delta:
                best_delta = delta
                best_file = fpath

        if best_file is None:
            continue

        # --- load topic-feature matrix ---
        Z = pd.read_csv(best_file, sep="\t", index_col=0)
        Z.index = Z.index.str.replace("#", "", regex=False)
        Z = Z.reindex(features)
        Z.columns = [f"Topic_{i}" for i in range(Z.shape[1])]
        Z_runs[fs].append(Z)

        # --- load topic-document matrix ---
        td_file = best_file.replace(f"_mRNA_topics.tsv.gz",f"_mRNA_topics_documents.tsv.gz")
        TD = pd.read_csv(td_file, sep="\t", index_col=0)
        TD.columns = [f"Topic_{i}" for i in range(TD.shape[1])]
        TD=TD-TD.mean(axis=0)
        TD_runs[fs].append(TD)

# ============================================================
# Dominant topic stability per cell type
# ============================================================

def dominant_topic_stability(Z_list, TD_list):
    sims = []

    for ct in celltypes.unique():

        # store top-k topic vectors PER RUN
        run_topics = []

        for Z, TD in zip(Z_list, TD_list):

            cells = celltypes[celltypes == ct].index
            cells = cells.intersection(TD.index)
            if len(cells) == 0:
                continue

            # P_centered(topic | celltype)
            p_ct = TD.loc[cells].mean(axis=0)

            # top-k topics
            top_topics = p_ct.nlargest(k).index

            vecs = [
                Z[t].fillna(0).values
                for t in top_topics
                if t in Z.columns
            ]

            if len(vecs) > 0:
                run_topics.append(vecs)

        # compare runs: max similarity across top-k topics
        for i, j in combinations(range(len(run_topics)), 2):

            sims_ij = [
                pdf_similarity(v1, v2)
                for v1 in run_topics[i]
                for v2 in run_topics[j]
            ]

            if len(sims_ij) > 0:
                sims.append(np.max(sims_ij))

    return np.nan if len(sims) == 0 else np.mean(sims)


outdir = f"Datasets/{dataset}/bionSBM"
os.makedirs(outdir, exist_ok=True)


rows = []

for fs in order:
    score = dominant_topic_stability(Z_runs[fs],TD_runs[fs])
    rows.append((fs, score))

res = pd.DataFrame(rows, columns=["Experiment", "Stability"])
res.to_csv(f"{outdir}/DominantTopicStability_top{k}_cosine_mRNA.tsv", sep="\t", index=False)

print("Done.")
