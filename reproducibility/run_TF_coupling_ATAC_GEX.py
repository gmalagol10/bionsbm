"""ATAC-TF/gene-topic coupling for the regulatory ablation study.

Experiments:
    Peak_mRNA_lncRNA: use mRNA topics
    Peak_mRNA:        use mRNA topics
    Peak_GEX:         use GEX topics restricted to the mRNA modality universe

For each ATAC/gene-topic pair, the HOMER-derived TF set remains fixed. The gene
topic is compared with random topics matched jointly on mean expression, cell
coverage and genomic gene length.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import pandas as pd

from helps import (
    GTF_PATHS, add_matching_strata, bh_fdr, find_best_topic_level,
    gene_covariates, iter_matched_null_topics,
    load_regulons, load_topic_gene_sets, read_gene_lookups,
)

DATASET = sys.argv[1]
N_PERM, N_RUNS, N_BINS, SEED = 1000, 25, 5, 0
MOTIF_ROOT = os.environ.get("MOTIF_ROOT", "Motifs_ablation")

EXPERIMENTS = {
    "Peak_mRNA_lncRNA": "mRNA",
    "Peak_mRNA": "mRNA",
    "Peak_GEX": "GEX",
}


def load_atac_tfs(dataset, exp, run, topic, regulons):
    """Read significant HOMER motifs and retain TFs represented in CollecTRI."""
    path = (
        f"Datasets/{dataset}/bionSBM/{MOTIF_ROOT}/{exp}/"
        f"Run{run}/Topic_{topic}/knownResults.txt"
    )
    if not os.path.isfile(path):
        return []

    try:
        motifs = pd.read_csv(path, sep="\t")
    except Exception:
        return []

    required = {"Motif Name", "q-value (Benjamini)"}
    if not required.issubset(motifs.columns):
        return []

    names = motifs.loc[motifs["q-value (Benjamini)"] < 0.05, "Motif Name"].dropna()
    tfs = [name.split("(")[0].strip() for name in names]
    return list(dict.fromkeys(
        tf for tf in tfs
        if not tf.lower().startswith("unknown") and tf in regulons
    ))


def score_pair(genes, tfs, regulons):
    """Normalized overlap between one gene topic and ATAC-derived TF regulons."""
    return sum(
        len(genes & regulons[tf]) / len(regulons[tf])
        for tf in tfs if regulons[tf]
    )


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
        peak_file = (
            f"{base_dir}/bionSBM_{exp}_level_{level}_Peak_topics.tsv.gz"
        )
        if not os.path.isfile(topic_file) or not os.path.isfile(peak_file):
            continue

        Z, topic_gene_sets = load_topic_gene_sets(topic_file)

        # read_gene_lookups() is based on the mRNA modality. Therefore, for
        # Peak_GEX, gene_covariates() automatically removes lncRNAs and keeps
        # only GEX genes belonging to the mRNA machinery universe.
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

        P = pd.read_csv(peak_file, sep="\t", index_col=0)
        atac_tf_sets = {
            topic: tfs
            for topic in range(len(P.columns))
            if (tfs := load_atac_tfs(DATASET, exp, run, topic, regulons))
        }
        if not atac_tf_sets:
            continue

        rng = np.random.default_rng(
            np.random.SeedSequence([SEED, exp_index, run])
        )

        for gene_topic, input_genes in topic_gene_sets.items():
            genes = set(input_genes) & set(stats.index)
            if not genes:
                continue

            null_scores = {
                atac_topic: np.empty(N_PERM) for atac_topic in atac_tf_sets
            }
            for i, random_genes in enumerate(
                iter_matched_null_topics(genes, stats, N_PERM, rng)
            ):
                for atac_topic, tfs in atac_tf_sets.items():
                    null_scores[atac_topic][i] = score_pair(
                        random_genes, tfs, regulons
                    )

            for atac_topic, tfs in atac_tf_sets.items():
                real = score_pair(genes, tfs, regulons)
                null = null_scores[atac_topic]
                mu, sigma = null.mean(), null.std()
                results.append({
                    "dataset": DATASET,
                    "exp": exp,
                    "gene_space": gene_space,
                    "run": run,
                    "level": level,
                    "ATAC_topic": atac_topic,
                    "GEX_topic": gene_topic,  # retained for plot compatibility
                    "n_TF": len(tfs),
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
    out_pairs, out_matrix = f"{out_dir}/{exp}_TF_coupling_ATAC_GEX_results.tsv", f"{out_dir}/{exp}_TF_coupling_ATAC_GEX_results_matrix.tsv"
    df.to_csv(out_pairs, sep="\t", index=False)
    matrix_columns = [
        "dataset", "exp", "gene_space", "run", "ATAC_topic", "GEX_topic",
        "zscore", "delta_score", "enrichment", "pval", "qval", "n_tests",
    ]
    df.reindex(columns=matrix_columns).to_csv(
        out_matrix, sep="\t", index=False
    )
    print(f"Saved: {out_pairs}\nSaved: {out_matrix}", flush=True)
