import os, sys, time
from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# USER INPUT
# ============================================================

dataset = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
n_runs = 25

main_fs = "Peak" if dataset not in ["BMMCCite", "Spleen"] else "ADT"
print(f"Dataset: {dataset} | Main FS: {main_fs} | top-k = {k}", flush=True)

order = [
    f"{main_fs}_mRNA_lncRNA",
    f"{main_fs}_mRNA",
    "mRNA_lncRNA",
]

# ============================================================
# Utilities
# ============================================================

def cosine_pdf_similarity(p, q):
    p = np.nan_to_num(p, nan=0.0)
    q = np.nan_to_num(q, nan=0.0)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    return cosine_similarity(p[None], q[None])[0, 0]


def selected_level(path):
    return os.path.basename(path).split("_level_", 1)[1].split("_", 1)[0]


def summarise(values, group_columns, covariate):
    if values.empty:
        return pd.DataFrame()
    grouped = values.groupby(group_columns, dropna=False)["CosineSimilarity"]
    out = grouped.agg(N="size", Mean="mean", Median="median", SD="std", Min="min", Max="max").reset_index()
    out["Range"] = out["Max"] - out["Min"]
    below = grouped.apply(lambda x: (x < 0.5).mean()).reset_index(name="FractionBelow0.5")
    out = out.merge(below, on=group_columns, how="left")
    out.insert(2, "Covariate", covariate)
    return out


# ============================================================
# Load metadata and aligned feature universe
# ============================================================

meta = pd.read_csv(f"Datasets/{dataset}/{dataset}_Metadata.tsv.gz", sep="\t", index_col=0)
celltypes = meta["CellType"].astype(str)
n_celltypes = celltypes.nunique()
features = sc.read_h5ad(f"Datasets/{dataset}/CM/{dataset}_mRNA_Def.h5ad").var.index

# ============================================================
# Load runs and retain run-level metadata
# ============================================================

runs = {fs: [] for fs in order}
manifest_rows = []

for run in range(n_runs):
    print(time.strftime("%H:%M:%S"), "Run", run, flush=True)

    for fs in order:
        base_dir = f"Datasets/{dataset}/bionSBM/bionSBM_{fs}/Runs/Run{run}"
        manifest = {"Dataset": dataset, "Experiment": fs, "Run": run, "BaseDir": base_dir}

        if not os.path.isdir(base_dir):
            manifest["Status"] = "missing_directory"
            manifest_rows.append(manifest)
            continue

        best_file, best_delta = None, np.inf
        for fname in os.listdir(base_dir):
            if not fname.endswith("_mRNA_topics.tsv.gz") or "_level_" not in fname:
                continue
            fpath = os.path.join(base_dir, fname)
            try:
                tmp = pd.read_csv(fpath, sep="\t", index_col=0, nrows=5)
            except Exception:
                continue
            delta = abs(tmp.shape[1] - n_celltypes)
            if delta < best_delta:
                best_file, best_delta = fpath, delta

        if best_file is None:
            manifest["Status"] = "no_topic_file"
            manifest_rows.append(manifest)
            continue

        Z = pd.read_csv(best_file, sep="\t", index_col=0)
        Z.index = Z.index.str.replace("#", "", regex=False)
        Z = Z.reindex(features)
        Z.columns = [f"Topic_{i}" for i in range(Z.shape[1])]

        td_file = best_file.replace(
            "_mRNA_topics.tsv.gz", "_mRNA_topics_documents.tsv.gz"
        )
        TD = pd.read_csv(td_file, sep="\t", index_col=0)
        TD.columns = [f"Topic_{i}" for i in range(TD.shape[1])]
        TD = TD - TD.mean(axis=0)

        level = selected_level(best_file)
        record = {
            "run": run,
            "level": level,
            "n_topics": Z.shape[1],
            "level_delta": best_delta,
            "Z": Z,
            "TD": TD,
        }
        runs[fs].append(record)
        manifest.update({
            "Status": "loaded",
            "SelectedLevel": level,
            "NTopics": Z.shape[1],
            "TopicCellTypeDifference": best_delta,
            "NTopicDocuments": TD.shape[0],
            "NFeatures": Z.shape[0],
            "TopicFile": best_file,
            "TopicDocumentFile": td_file,
        })
        manifest_rows.append(manifest)

# ============================================================
# Original stability plus detailed pairwise data
# ============================================================

summary_rows, topic_rows, pair_rows = [], [], []

for fs in order:
    all_similarities = []

    for ct in celltypes.unique():
        ct_cells = celltypes[celltypes == ct].index
        selected = []

        for record in runs[fs]:
            Z, TD = record["Z"], record["TD"]
            cells = ct_cells.intersection(TD.index)
            if len(cells) == 0:
                continue

            p_ct = TD.loc[cells].mean(axis=0)
            top_topics = p_ct.nlargest(k)
            top_two = p_ct.nlargest(min(2, len(p_ct)))
            margin = top_two.iloc[0] - top_two.iloc[1] if len(top_two) > 1 else np.nan

            names, ranks, activities, vectors = [], [], [], []
            for rank, (topic, activity) in enumerate(top_topics.items(), start=1):
                if topic not in Z.columns:
                    continue
                names.append(topic)
                ranks.append(rank)
                activities.append(activity)
                vectors.append(Z[topic].fillna(0).values)
                topic_rows.append({
                    "Dataset": dataset,
                    "Experiment": fs,
                    "Run": record["run"],
                    "CellType": ct,
                    "Rank": rank,
                    "Topic": topic,
                    "TopicActivity": activity,
                    "Top1Top2Margin": margin,
                    "NCells": len(cells),
                    "SelectedLevel": record["level"],
                    "NTopics": record["n_topics"],
                    "TopicCellTypeDifference": record["level_delta"],
                })

            if vectors:
                selected.append({
                    **record,
                    "names": names,
                    "ranks": ranks,
                    "activities": activities,
                    "vectors": vectors,
                    "margin": margin,
                    "n_cells": len(cells),
                })

        for a, b in combinations(selected, 2):
            similarities = np.array([
                cosine_pdf_similarity(v1, v2)
                for v1 in a["vectors"]
                for v2 in b["vectors"]
            ])
            if similarities.size == 0:
                continue

            best = int(np.argmax(similarities))
            i, j = np.unravel_index(best, (len(a["vectors"]), len(b["vectors"])))
            similarity = float(similarities[best])
            all_similarities.append(similarity)

            level_pair = "|".join(sorted([str(a["level"]), str(b["level"])]))
            topic_pair = "|".join(map(str, sorted([a["n_topics"], b["n_topics"]])))
            run_pair = "|".join(map(str, sorted([a["run"], b["run"]])))
            rank_pair = "|".join(map(str, sorted([a["ranks"][i], b["ranks"][j]])))
            margins = np.array([a["margin"], b["margin"]], dtype=float)
            mean_margin = float(np.nanmean(margins)) if np.isfinite(margins).any() else np.nan
            pair_rows.append({
                "Dataset": dataset,
                "Experiment": fs,
                "CellType": ct,
                "Run1": a["run"],
                "Run2": b["run"],
                "RunPair": run_pair,
                "CosineSimilarity": similarity,
                "MatchedTopicRun1": a["names"][i],
                "MatchedTopicRun2": b["names"][j],
                "MatchedRankRun1": a["ranks"][i],
                "MatchedRankRun2": b["ranks"][j],
                "MatchedRankPair": rank_pair,
                "MatchedActivityRun1": a["activities"][i],
                "MatchedActivityRun2": b["activities"][j],
                "Top1Top2MarginRun1": a["margin"],
                "Top1Top2MarginRun2": b["margin"],
                "MeanTop1Top2Margin": mean_margin,
                "NCellsRun1": a["n_cells"],
                "NCellsRun2": b["n_cells"],
                "MeanNCells": np.mean([a["n_cells"], b["n_cells"]]),
                "LevelRun1": a["level"],
                "LevelRun2": b["level"],
                "LevelPair": level_pair,
                "SameLevel": a["level"] == b["level"],
                "NTopicsRun1": a["n_topics"],
                "NTopicsRun2": b["n_topics"],
                "NTopicsPair": topic_pair,
                "TopicCountDifference": abs(a["n_topics"] - b["n_topics"]),
                "TopicCellTypeDifferenceRun1": a["level_delta"],
                "TopicCellTypeDifferenceRun2": b["level_delta"],
            })

    summary_rows.append((fs, np.mean(all_similarities) if all_similarities else np.nan))

# ============================================================
# Save original and detailed outputs
# ============================================================

outdir = f"Datasets/{dataset}/bionSBM"
os.makedirs(outdir, exist_ok=True)
base = f"{outdir}/DominantTopicStability_top{k}_cosine_mRNA"

# Original output: same filename and columns as before.
pd.DataFrame(summary_rows, columns=["Experiment", "Stability"]).to_csv(
    f"{base}.tsv", sep="\t", index=False
)

manifest = pd.DataFrame(manifest_rows)
topics = pd.DataFrame(topic_rows)
pairs = pd.DataFrame(pair_rows)
manifest.to_csv(f"{base}_run_manifest.tsv.gz", sep="\t", index=False)
topics.to_csv(f"{base}_dominant_topics.tsv.gz", sep="\t", index=False)
pairs.to_csv(f"{base}_pairwise.tsv.gz", sep="\t", index=False)

stratified = []
if not pairs.empty:
    celltype_values = pairs[["Dataset", "Experiment", "CellType", "CosineSimilarity"]].rename(columns={"CellType": "Group"})
    stratified.append(summarise(celltype_values, ["Dataset", "Experiment", "Group"], "CellType"))

    by_run = pd.concat([
        pairs[["Dataset", "Experiment", "Run1", "CosineSimilarity"]].rename(columns={"Run1": "Group"}),
        pairs[["Dataset", "Experiment", "Run2", "CosineSimilarity"]].rename(columns={"Run2": "Group"}),
    ], ignore_index=True)
    stratified.append(summarise(by_run, ["Dataset", "Experiment", "Group"], "Run"))

    for column in ["RunPair", "LevelPair", "SameLevel", "NTopicsPair", "MatchedRankPair"]:
        values = pairs[["Dataset", "Experiment", column, "CosineSimilarity"]].rename(columns={column: "Group"})
        stratified.append(summarise(values, ["Dataset", "Experiment", "Group"], column))

stratified = pd.concat([x for x in stratified if not x.empty], ignore_index=True, sort=False) if stratified else pd.DataFrame()
stratified.to_csv(f"{base}_stratified_summary.tsv.gz", sep="\t", index=False)

print("Done.")
