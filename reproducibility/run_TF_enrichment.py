import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import pandas as pd

from helps import (
    GTF_PATHS, add_matching_strata, bh_fdr, find_best_topic_level, gene_covariates, iter_matched_null_topics,
    load_regulons, load_topic_gene_sets, read_gene_lookups,
)

DATASET = sys.argv[1]
N_PERM, N_RUNS, N_BINS, SEED = 1000, 25, 5, 0

EXPERIMENTS_PEAK = {
    "Peak_mRNA_lncRNA": "mRNA",
    "Peak_mRNA": "mRNA",
    "Peak_GEX": "GEX",
}

EXPERIMENTS_ADT = {
    "ADT_mRNA_lncRNA": "mRNA",
    "ADT_mRNA": "mRNA",
    "ADT_GEX": "GEX",
}

EXPERIMENTS = EXPERIMENTS_PEAK if DATASET not in ["BMMCCite", "Spleen"] else EXPERIMENTS_ADT
print("Dataset", DATASET, flush=True)
print("Experiments", EXPERIMENTS, flush=True)

def score_topic(genes, regulons):
    """Original aggregate overlap between topic genes and all TF regulons."""
    return sum(len(genes & targets) for targets in regulons.values())


if GTF_PATHS.get(DATASET) in {None, "FILL_ME"}:
    raise ValueError(f"Fill GTF_PATHS['{DATASET}'] in helps.py")

meta = pd.read_csv(
    f"Datasets/{DATASET}/{DATASET}_Metadata.tsv.gz", sep="\t", index_col=0
)
n_celltypes = meta["CellType"].nunique()
expression_lookup, gtf_lookup = read_gene_lookups(DATASET)
regulons = load_regulons(DATASET)

for exp_index, (exp, gene_space) in enumerate(EXPERIMENTS.items()):
    results = []

    for run in range(N_RUNS):
        print(f"{DATASET} | {exp} | Run {run}", flush=True)
        base_dir = f"Datasets/{DATASET}/bionSBM/bionSBM_{exp}/Runs/Run{run}"
        if not os.path.isdir(base_dir):
            continue

        level = find_best_topic_level(base_dir, n_celltypes)
        if level is None:
            continue

        topic_file = (
            f"{base_dir}/bionSBM_{exp}_level_{level}_{gene_space}_topics.tsv.gz"
        )
        if not os.path.isfile(topic_file):
            continue

        Z, topic_gene_sets = load_topic_gene_sets(topic_file)

        # Expression lookup comes only from the mRNA modality. In Peak_GEX this
        # explicitly restricts scoring and null sampling to the mRNA universe.
        stats = add_matching_strata(
            gene_covariates(Z.index.unique(), expression_lookup, gtf_lookup),
            N_BINS,
        )
        print(
            f"  mRNA-matchable genes: {len(stats)}/{Z.index.nunique()}",
            flush=True,
        )
        if stats.empty:
            continue

        rng = np.random.default_rng(
            np.random.SeedSequence([SEED, exp_index, run])
        )

        for topic, input_genes in topic_gene_sets.items():
            genes = set(input_genes) & set(stats.index)
            if not genes:
                continue

            real = score_topic(genes, regulons)
            null = np.fromiter(
                (
                    score_topic(random_genes, regulons)
                    for random_genes in iter_matched_null_topics(
                        genes, stats, N_PERM, rng
                    )
                ),
                dtype=float,
                count=N_PERM,
            )
            mu, sigma = null.mean(), null.std()
            results.append({
                "dataset": DATASET,
                "exp": exp,
                "gene_space": gene_space,
                "run": run,
                "level": level,
                "topic": topic,
                "n_genes_input": len(input_genes),
                "n_genes_mRNA_universe": len(genes),
                "matching_fraction": len(genes) / len(input_genes),
                "real_score": real,
                "null_mean": mu,
                "null_std": sigma,
                "delta_score": real - mu,
                "zscore": (real - mu) / sigma if sigma > 0 else np.nan,
                "enrichment": real / mu if mu > 0 else np.nan,
                "pval": (np.sum(null >= real) + 1) / (N_PERM + 1)})

        df = pd.DataFrame(results)
        if not df.empty:
            df["qval"] = df.groupby("run")["pval"].transform(
                lambda x: bh_fdr(x.to_numpy())
            )
            df["n_tests"] = df.groupby("run")["pval"].transform("size")
        out_dir = f"Datasets/{DATASET}/bionSBM/bionSBM_{exp}"
        os.makedirs(out_dir, exist_ok=True)
        output = f"{out_dir}/{exp}_TF_enrichment_results.tsv"
        df.to_csv(output, sep="\t", index=False)
        print(f"Saved: {output}", flush=True)
