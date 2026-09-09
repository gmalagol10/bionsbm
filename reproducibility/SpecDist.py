import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["PBMC", "MouseSkin", "BMMCMultiOme", "HSPC", "BMMCCite", "Spleen"]
ATAC_DATASETS = DATASETS[:4]
ADT_DATASETS = DATASETS[4:]
RUNS = range(25)

COLUMNS = [
    "Dataset", "Exp", "Run", "FS", "CT", "Representation",
    "DifferentTopics", "NumberOfCt", "NumberOfTopics",
    "Topics_distinctiveness",
]


def specificity_rows(scores, meta, dataset, exp, run, fs, representation):
    """Apply the original topic/factor specificity calculation."""
    scores = scores.loc[
        ~scores.index.duplicated() & scores.index.isin(meta.index)
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if scores.empty:
        return []

    means = scores.assign(CT=meta.loc[scores.index, "CellType"]).dropna()
    means = means.groupby("CT").mean()

    # Original normalization: scale each topic/factor across cell types.
    span = means.max() - means.min()
    means = ((means - means.min()) / span.replace(0, np.nan)).dropna(
        axis=1, how="all"
    )
    if means.empty:
        return []

    winners = means.idxmax(axis=1).dropna()
    n_distinct = winners.nunique()
    rows = []

    for cell_type, winner in winners.items():
        others = means.columns[means.columns != winner]
        distinctiveness = (
            means.loc[cell_type, winner] - means.loc[cell_type, others].mean()
            if len(others) else np.nan
        )
        rows.append([
            dataset, exp, run, fs, cell_type, representation,
            n_distinct, means.shape[0], means.shape[1], distinctiveness,
        ])

    return rows


def save(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        path, sep="\t", compression="gzip", index=False
    )


# ------------------------------- bionSBM -------------------------------

for dataset in DATASETS:
    meta = pd.read_csv(
        f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
        sep="\t", index_col=0,
    )
    prefix = "ADT" if dataset in ADT_DATASETS else "Peak"
    exps = [
        f"bionSBM_{prefix}",
        "bionSBM_mRNA",
        f"bionSBM_{prefix}_GEX",
        f"bionSBM_{prefix}_mRNA",
        f"bionSBM_{prefix}_lncRNA",
        f"bionSBM_{prefix}_mRNA_lncRNA",
    ]
    rows = []

    for exp in exps:
        for run in RUNS:
            for fs in exp.split("_")[1:]:
                levels = {}

                for level in range(7):
                    file = (
                        f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/"
                        f"{exp}_level_{level}_{fs}_topics_documents.tsv.gz"
                    )
                    if os.path.isfile(file):
                        n_topics = pd.read_csv(
                            file, sep="\t", index_col=0, nrows=0
                        ).shape[1]
                        levels[level] = abs(
                            n_topics - meta["CellType"].nunique()
                        )

                if not levels:
                    continue

                level = min(levels, key=levels.get)
                file = (
                    f"Datasets/{dataset}/bionSBM/{exp}/Runs/Run{run}/"
                    f"{exp}_level_{level}_{fs}_topics_documents.tsv.gz"
                )
                scores = pd.read_csv(file, sep="\t", index_col=0)

                rows += specificity_rows(
                    scores, meta, dataset, exp, run, fs,
                    f"Topic_Level_{level}",
                )

    save(
        rows,
        f"Datasets/{dataset}/bionSBM/"
        f"{dataset}_bionSBM_TopicsSpecDist.tsv.gz",
    )


# ------------------------------ ShareTopic -----------------------------

for dataset in ATAC_DATASETS:
    meta = pd.read_csv(
        f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
        sep="\t", index_col=0,
    )
    exps = [
        "ShareTopic_Peak_GEX",
        "ShareTopic_Peak_mRNA",
        "ShareTopic_Peak_lncRNA",
    ]
    rows = []

    for exp in exps:
        for run in RUNS:
            file = (
                f"Datasets/{dataset}/ShareTopic/{exp}/Runs/Run{run}/"
                f"{dataset}_{exp}_Topic_Cell.tsv.gz"
            )
            if os.path.isfile(file):
                scores = pd.read_csv(file, sep="\t", index_col=0).T
                rows += specificity_rows(
                    scores, meta, dataset, exp, run, "Joint", "Topic"
                )

    save(
        rows,
        f"Datasets/{dataset}/ShareTopic/"
        f"{dataset}_ShareTopic_TopicsSpecDist.tsv.gz",
    )


# ------------------------------- Mowgli --------------------------------

for dataset in DATASETS:
    meta = pd.read_csv(
        f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
        sep="\t", index_col=0,
    )
    prefix = "ADT" if dataset in ADT_DATASETS else "Peak"
    exps = [
        f"Mowgli_{prefix}_GEX",
        f"Mowgli_{prefix}_mRNA",
        f"Mowgli_{prefix}_lncRNA",
        f"Mowgli_{prefix}_mRNA_lncRNA",
        "Mowgli_mRNA_lncRNA",
    ]
    rows = []

    for exp in exps:
        for run in RUNS:
            file = (
                f"Datasets/{dataset}/Mowgli/{exp}/Runs/Run{run}/"
                f"{dataset}_{exp}_Embedding.tsv.gz"
            )
            if os.path.isfile(file):
                scores = pd.read_csv(file, sep="\t", index_col=0).T
                rows += specificity_rows(
                    scores, meta, dataset, exp, run, "Joint", "Topic"
                )

    save(
        rows,
        f"Datasets/{dataset}/Mowgli/"
        f"{dataset}_Mowgli_TopicsSpecDist.tsv.gz",
    )


# ------------------------------- COBOLT --------------------------------
# COBOLT embeddings are latent category scores (dimensions x cells).
# Treat them exactly like Mowgli: transpose to cells x factors and apply
# the same specificity/distinctiveness calculation.

for dataset in DATASETS:
    meta = pd.read_csv(
        f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
        sep="\t", index_col=0,
    )
    prefix = "ADT" if dataset in ADT_DATASETS else "Peak"
    exps = [
        f"COBOLT_{prefix}_GEX",
        f"COBOLT_{prefix}_mRNA",
        f"COBOLT_{prefix}_lncRNA",
        f"COBOLT_{prefix}_mRNA_lncRNA",
        "COBOLT_mRNA_lncRNA",
    ]
    rows = []

    for exp in exps:
        for run in RUNS:
            file = (
                f"Datasets/{dataset}/COBOLT/{exp}/Runs/Run{run}/"
                f"{dataset}_{exp}_Embedding.tsv.gz"
            )
            if os.path.isfile(file):
                scores = pd.read_csv(file, sep="\t", index_col=0).T
                rows += specificity_rows(
                    scores, meta, dataset, exp, run, "Joint", "Factor"
                )

    save(
        rows,
        f"Datasets/{dataset}/COBOLT/"
        f"{dataset}_COBOLT_FactorsSpecDist.tsv.gz",
    )


# ------------------------------- MOFA+ ---------------------------------
# MOFA factors are signed. Use their absolute scores so factor assignment
# depends on magnitude and is invariant to sign flips.

for dataset in DATASETS:
    meta = pd.read_csv(
        f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz",
        sep="\t", index_col=0,
    )
    prefix = "ADT" if dataset in ADT_DATASETS else "Peak"
    exps = [
        f"MOFA_{prefix}_GEX",
        f"MOFA_{prefix}_mRNA",
        f"MOFA_{prefix}_lncRNA",
        f"MOFA_{prefix}_mRNA_lncRNA",
        "MOFA_mRNA_lncRNA",
    ]
    rows = []

    for exp in exps:
        for run in RUNS:
            file = (
                f"Datasets/{dataset}/MOFA/{exp}/Runs/Run{run}/"
                f"{dataset}_{exp}_Embedding.tsv.gz"
            )
            if not os.path.isfile(file):
                continue

            factors = pd.read_csv(file, sep="\t", index_col=0).T
            factors = factors.loc[
                ~factors.index.duplicated() & factors.index.isin(meta.index)
            ].replace([np.inf, -np.inf], np.nan).dropna()

            if factors.empty:
                continue

            # MOFA factors are signed; use absolute scores so each cell is
            # associated with the factor having the strongest magnitude.
            scores = factors.abs()

            rows += specificity_rows(
                scores, meta, dataset, exp, run, "Joint", "FactorAbs"
            )

    save(
        rows,
        f"Datasets/{dataset}/MOFA/"
        f"{dataset}_MOFA_FactorsSpecDist.tsv.gz",
    )
