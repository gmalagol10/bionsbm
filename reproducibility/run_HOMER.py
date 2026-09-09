"""Run HOMER motif enrichment for all Peak/RNA ablation experiments.

The same dominant-topic workflow is applied independently to:
Peak_mRNA_lncRNA, Peak_mRNA, Peak_lncRNA and Peak_GEX.

Results are written below ``Motifs_ablation`` by default, preserving the original
HOMER results. Set MOTIF_ROOT to a different folder name for another replicate.
Existing completed topic folders are skipped rather than overwritten.
"""
import shutil
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scanpy as sc

DATASET = sys.argv[1]
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_RUNS = 25
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
MOTIF_ROOT = os.environ.get("MOTIF_ROOT", "Motifs_ablation")

EXPERIMENTS = (
    "Peak_mRNA_lncRNA",
    "Peak_mRNA",
    "Peak_lncRNA",
    "Peak_GEX",
)
GENOMES = {
    "PBMC": "../AnnotRef/hs/T2T/chm13v2.0.fa",
    "HSPC": "hg38",
    "MouseSkin": "mm10",
    "BMMCMultiOme": "hg38",
}


def select_best_peak_level(base_dir, n_celltypes):
    """Select the Peak-topic matrix whose topic count is closest to cell types."""
    best_file, best_delta = None, np.inf
    for name in os.listdir(base_dir):
        if not name.endswith("_Peak_topics.tsv.gz") or "_level_" not in name:
            continue
        path = os.path.join(base_dir, name)
        try:
            n_topics = pd.read_csv(path, sep="\t", index_col=0, nrows=1).shape[1]
        except Exception:
            continue
        delta = abs(n_topics - n_celltypes)
        if delta < best_delta:
            best_file, best_delta = path, delta
    return best_file


def parse_peak(name):
    """Parse common ``chr_start_end``, ``chr:start-end`` and related formats."""
    name = str(name).replace("#", "")
    match = re.match(r"^(.+?)[_:](\d+)[_-](\d+)$", name)
    if not match:
        match = re.match(r"^(.+?)-(\d+)-(\d+)$", name)
    if not match:
        return None
    chrom, start, end = match.groups()
    return chrom, int(start), int(end)

def homer_result_complete(topic_dir):
    known_results = os.path.join(topic_dir, "knownResults.txt")

    if not os.path.isfile(known_results):
        return False

    if os.path.getsize(known_results) == 0:
        return False

    try:
        result = pd.read_csv(known_results, sep="\t")
    except Exception:
        return False

    required = {"Motif Name", "q-value (Benjamini)"}
    return required.issubset(result.columns)
    
def run_motif(topic, peaks, run, exp, genome, out_root):
    topic_dir = os.path.join(out_root, f"Run{run}", topic)

    if homer_result_complete(topic_dir):
        print(
            f"    SKIP completed | {exp} | Run {run} | {topic}",
            flush=True,
        )
        return

    # Remove output from a previously interrupted HOMER invocation.
    if os.path.isdir(topic_dir):
        print(
            f"    RESTART incomplete | {exp} | Run {run} | {topic}",
            flush=True,
        )
        shutil.rmtree(topic_dir)

    intervals = [
        parsed
        for peak in peaks
        if (parsed := parse_peak(peak))
    ]

    if not intervals:
        print(
            f"    SKIP no parseable peaks | {exp} | Run {run} | {topic}",
            flush=True,
        )
        return

    os.makedirs(topic_dir, exist_ok=True)

    peak_file = os.path.join(topic_dir, "peaks.tsv")
    pd.DataFrame(intervals).to_csv(
        peak_file,
        sep="\t",
        header=False,
        index=False,
    )

    print(
        f"    HOMER | {exp} | Run {run} | {topic} | "
        f"{len(intervals)} peaks",
        flush=True,
    )

    subprocess.run(
        ["findMotifsGenome.pl", peak_file, genome, topic_dir, "-p", "1"],
        check=True,
    )


if DATASET not in GENOMES:
    raise ValueError(f"No HOMER genome configured for {DATASET}: {sorted(GENOMES)}")

meta = pd.read_csv(
    f"Datasets/{DATASET}/{DATASET}_Metadata.tsv.gz", sep="\t", index_col=0
)
celltypes = meta["CellType"].astype(str)
n_celltypes = celltypes.nunique()
peak_universe = sc.read_h5ad(
    f"Datasets/{DATASET}/CM/{DATASET}_Peak_Def.h5ad"
).var_names.astype(str)

print(f"Dataset: {DATASET} | top-k: {TOP_K} | jobs: {N_JOBS}", flush=True)

for exp in EXPERIMENTS:
    out_root = f"Datasets/{DATASET}/bionSBM/{MOTIF_ROOT}/{exp}"
    os.makedirs(out_root, exist_ok=True)
    print(f"\n===== Experiment: {exp} =====", flush=True)

    for run in range(N_RUNS):
        print(f"  Run {run}", flush=True)
        base_dir = f"Datasets/{DATASET}/bionSBM/bionSBM_{exp}/Runs/Run{run}"
        if not os.path.isdir(base_dir):
            continue

        peak_file = select_best_peak_level(base_dir, n_celltypes)
        if peak_file is None:
            continue

        Z = pd.read_csv(peak_file, sep="\t", index_col=0)
        Z.index = Z.index.astype(str).str.replace("#", "", regex=False)
        Z = Z.reindex(peak_universe)
        Z.columns = [f"Topic_{i}" for i in range(Z.shape[1])]

        td_file = peak_file.replace(
            "_Peak_topics.tsv.gz", "_Peak_topics_documents.tsv.gz"
        )
        if not os.path.isfile(td_file):
            continue
        TD = pd.read_csv(td_file, sep="\t", index_col=0)
        TD.columns = Z.columns
        TD -= TD.mean(axis=0)

        dominant = set()
        for celltype in celltypes.unique():
            cells = celltypes[celltypes == celltype].index.intersection(TD.index)
            if len(cells):
                dominant.update(TD.loc[cells].mean(axis=0).nlargest(TOP_K).index)

        print(f"    Dominant topics: {len(dominant)}", flush=True)
        jobs = []
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            for topic in sorted(dominant):
                peaks = Z.index[Z[topic].fillna(0) > 0].tolist()
                jobs.append(executor.submit(
                    run_motif, topic, peaks, run, exp, GENOMES[DATASET], out_root
                ))
            for job in as_completed(jobs):
                job.result()

print("\nAll HOMER ablation runs completed.", flush=True)
