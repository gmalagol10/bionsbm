#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

ROOT = Path("Datasets/GSE268630")
OUT = ROOT / "GSE268630_GEX_processed.h5ad"
CALLED_BARCODES = ROOT / "GSE268630_GEX_called_barcodes.txt"
PROCESSED_BARCODES = ROOT / "GSE268630_GEX_processed_barcodes.txt"


def sample_id(path):
	return path.name.removeprefix("GSE268630_").removesuffix("_filtered_feature_bc_matrix.h5")


def log_quantile(values, q):
	values = np.asarray(values, dtype=float)
	return 10 ** np.quantile(np.log10(values[values > 0]), q)


files = sorted(ROOT.glob("GSE268630_*_filtered_feature_bc_matrix.h5"))
if len(files) != 25:
	raise RuntimeError(f"Expected 25 H5 files in {ROOT}, found {len(files)}")

adatas = []
for path in files:
	sample = sample_id(path)
	print(f"Reading {sample}", flush=True)
	x = sc.read_10x_h5(path, gex_only=False)
	x.var_names_make_unique()
	if "feature_types" not in x.var:
		raise KeyError(f"{path} has no var['feature_types']")
	x = x[:, x.var["feature_types"] == "Gene Expression"].copy()
	x.obs["barcode"] = x.obs_names.astype(str)
	x.obs["sample"] = sample
	x.obs_names = x.obs["barcode"] + "#" + sample
	adatas.append(x)

adata = ad.concat(adatas, join="inner", merge="first", index_unique=None)
adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
adata.obs_names.to_series().to_csv(CALLED_BARCODES, index=False, header=False)
print("Concatenated shape:", adata.shape, flush=True)

# QC equivalent to helps.py: 5th–95th log-scale quantiles for cells/features,
# mitochondrial filtering at the 90th percentile, and doublet removal.
adata.var["MT"] = adata.var_names.str.startswith(("MT", "mt"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)

min_features = log_quantile(adata.obs["n_genes_by_counts"], 0.05)
max_features = log_quantile(adata.obs["n_genes_by_counts"], 0.95)
min_cells = log_quantile(adata.var["n_cells_by_counts"], 0.05)
max_cells = log_quantile(adata.var["n_cells_by_counts"], 0.95)

adata = adata[
	adata.obs["n_genes_by_counts"].between(min_features, max_features),
	adata.var["n_cells_by_counts"].between(min_cells, max_cells),
].copy()
print("After cell/feature QC:", adata.shape, flush=True)

max_mt = np.quantile(adata.obs["pct_counts_MT"].dropna(), 0.9)
adata = adata[adata.obs["pct_counts_MT"] <= max_mt].copy()
print("After mitochondrial QC:", adata.shape, flush=True)

# Scrublet is run per library rather than across all donors.
adata.obs["doublet_score"] = np.nan
adata.obs["predicted_doublet"] = False
for sample in adata.obs["sample"].unique():
	idx = adata.obs["sample"] == sample
	tmp = adata[idx].copy()
	print(f"Scrublet: {sample} ({tmp.n_obs} cells)", flush=True)
	sc.pp.scrublet(tmp)
	adata.obs.loc[tmp.obs_names, "doublet_score"] = tmp.obs["doublet_score"]
	adata.obs.loc[tmp.obs_names, "predicted_doublet"] = tmp.obs["predicted_doublet"]

adata = adata[~adata.obs["predicted_doublet"]].copy()
print("After doublet filtering:", adata.shape, flush=True)

# Select the upper 10% of genes by Seurat-v3 normalized variance.
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", subset=False)
cutoff = np.nanquantile(adata.var["variances_norm"], 0.9)
adata = adata[:, adata.var["variances_norm"] > cutoff].copy()
adata.layers["raw"] = adata.X.copy()
print("After HVG filtering:", adata.shape, flush=True)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

cell_mask = np.asarray((adata.X != 0).sum(axis=1)).ravel() > 0
feature_mask = np.asarray((adata.X != 0).sum(axis=0)).ravel() > 0
adata = adata[cell_mask, feature_mask].copy()

adata.uns["processing"] = {
	"X": "library-size normalized and log1p transformed",
	"layers/raw": "filtered integer UMI counts",
	"cell_filter": "5th-95th log10 n_genes quantiles; <=90th percentile mitochondrial percentage; Scrublet singlets",
	"feature_filter": "5th-95th log10 n_cells quantiles; upper 10% Seurat-v3 normalized variance",
}
adata.obs_names.to_series().to_csv(PROCESSED_BARCODES, index=False, header=False)
adata.write_h5ad(OUT, compression="gzip")
print(f"Written {OUT}: {adata.shape}", flush=True)
