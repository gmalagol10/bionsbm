import os, sys, time, subprocess
import numpy as np
import pandas as pd
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# USER INPUT
# ============================================================

dataset = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 1

n_runs = 25
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

fs = "Peak_mRNA_lncRNA"
out_root = f"Datasets/{dataset}/bionSBM/Motifs/{fs}"
os.makedirs(out_root, exist_ok=True)

ds_infos = pd.DataFrame(
    [["PBMC","HSPC","MouseSkin","BMMCMultiOme"],
     ["../AnnotRef/hs/T2T/chm13v2.0.fa", "hg38", "mm10", "hg38"]],
    index=["Dataset","Genome"]
).T

genome = ds_infos.loc[ds_infos["Dataset"] == dataset, "Genome"].item()

print(f"Dataset: {dataset} | Experiment: {fs} | top-k = {k}", flush=True)
print(f"Parallel motif jobs: {n_jobs}", flush=True)

# ============================================================
# Load metadata
# ============================================================

meta = pd.read_csv(
    f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
    sep="\t",
    index_col=0
)

celltypes = meta["CellType"].astype(str)
n_celltypes = celltypes.nunique()

features = sc.read_h5ad(
    f"Datasets/{dataset}/CM/{dataset}_Peak_Def.h5ad"
).var.index

# ============================================================
# Utilities
# ============================================================

def select_best_level(base_dir):
    best_file = None
    best_delta = np.inf

    for fname in os.listdir(base_dir):
        if not fname.endswith("_Peak_topics.tsv.gz"):
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

    return best_file

# ============================================================
# Worker: motif analysis for ONE topic
# ============================================================

def run_motif_for_topic(topic, Z, run, genome):

    peaks = Z[topic].fillna(0)
    peaks = peaks[peaks > 0].index

    if len(peaks) == 0:
        return

    topic_dir = os.path.join(out_root, f"Run{run}", topic)
    os.makedirs(topic_dir, exist_ok=True)

    peak_file = os.path.join(topic_dir, "peaks.tsv")
    pd.DataFrame([p.split("_") for p in peaks]).to_csv(
        peak_file, sep="\t", header=None, index=False
    )

    print(
        f"    RUN MOTIF | Run {run} | {topic} | {len(peaks)} peaks",
        flush=True
    )

    cmd = (
        f"findMotifsGenome.pl {peak_file} {genome} {topic_dir} -p 1"
    )
    os.system(cmd)

# ============================================================
# Main loop over runs (SERIAL)
# ============================================================

for run in range(n_runs):

    print(f"\n===== Run {run} =====", flush=True)

    base_dir = (
        f"Datasets/{dataset}/bionSBM/"
        f"bionSBM_{fs}/Runs/Run{run}"
    )

    if not os.path.isdir(base_dir):
        print("  → run directory missing, skipping", flush=True)
        continue

    best_file = select_best_level(base_dir)
    if best_file is None:
        print("  → no valid level found, skipping", flush=True)
        continue

    print("  Using:", os.path.basename(best_file), flush=True)

    # --------------------------------------------------------
    # Load topic–peak matrix (Z)
    # --------------------------------------------------------

    Z = pd.read_csv(best_file, sep="\t", index_col=0)
    Z.index = Z.index.str.replace("#", "", regex=False)
    Z = Z.reindex(features)
    Z.columns = [f"Topic_{i}" for i in range(Z.shape[1])]

    # --------------------------------------------------------
    # Load topic–cell matrix (TD)
    # --------------------------------------------------------

    td_file = best_file.replace(
        "_Peak_topics.tsv.gz",
        "_Peak_topics_documents.tsv.gz"
    )

    TD = pd.read_csv(td_file, sep="\t", index_col=0)
    TD.columns = Z.columns
    TD = TD - TD.mean(axis=0)

    # ========================================================
    # Identify dominant topics
    # ========================================================

    dominant_topics = set()

    for ct in celltypes.unique():
        cells = celltypes[celltypes == ct].index
        cells = cells.intersection(TD.index)
        if len(cells) == 0:
            continue

        p_ct = TD.loc[cells].mean(axis=0)
        dominant_topics.update(p_ct.nlargest(k).index)

    dominant_topics = sorted(dominant_topics)
    print(f"  Dominant topics: {len(dominant_topics)}", flush=True)

    # ========================================================
    # Parallel motif analysis (TOPIC LEVEL)
    # ========================================================

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                run_motif_for_topic,
                topic, Z, run, genome
            )
            for topic in dominant_topics
        ]

        for _ in as_completed(futures):
            pass

print("\nAll runs completed.", flush=True)

