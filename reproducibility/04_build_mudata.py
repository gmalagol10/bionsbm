#!/usr/bin/env python3
from pathlib import Path
import anndata as ad
import muon as mu

ROOT = Path("Datasets/GSE268630")
GEX = ROOT / "GSE268630_GEX_processed.h5ad"
ATAC = ROOT / "GSE268630_ATAC_processed.h5ad"
OUT = ROOT / "GSE268630_GEX_ATAC_processed.h5mu"


def canonicalize_cells(adata, modality):
    required = {"barcode", "sample"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"{modality} is missing obs columns: {sorted(missing)}")

    barcode = adata.obs["barcode"].astype(str)
    sample = adata.obs["sample"].astype(str)

    adata.obs_names = barcode + "#" + sample

    if not adata.obs_names.is_unique:
        duplicated = adata.obs_names[adata.obs_names.duplicated()].unique()
        raise RuntimeError(
            f"{modality} still has duplicated canonical IDs; "
            f"examples: {list(duplicated[:5])}"
        )

    return adata


gex = canonicalize_cells(ad.read_h5ad(GEX), "GEX")
atac = canonicalize_cells(ad.read_h5ad(ATAC), "ATAC")

print("GEX cells:", gex.n_obs)
print("ATAC cells:", atac.n_obs)
print("GEX samples:", sorted(gex.obs["sample"].astype(str).unique()))
print("ATAC samples:", sorted(atac.obs["sample"].astype(str).unique()))

common = gex.obs_names.intersection(atac.obs_names, sort=False)
print("Common cells:", len(common))

if len(common) == 0:
    gex_samples = set(gex.obs["sample"].astype(str))
    atac_samples = set(atac.obs["sample"].astype(str))

    raise RuntimeError(
        "No matching canonical cell IDs.\n"
        f"Samples only in GEX: {sorted(gex_samples - atac_samples)}\n"
        f"Samples only in ATAC: {sorted(atac_samples - gex_samples)}"
    )

gex = gex[common].copy()
atac = atac[common].copy()

assert gex.obs_names.equals(atac.obs_names)

mdata = mu.MuData({
    "GEX": gex,
    "Peak": atac,
})

mdata.obs["barcode"] = gex.obs["barcode"].astype(str).values
mdata.obs["sample"] = gex.obs["sample"].astype(str).values

mdata.uns["processing"] = {
    "cell_selection": "intersection of cells passing GEX and ATAC QC",
    "cell_id": "barcode#sample",
    "GEX_X": "library-size normalized and log1p transformed",
    "GEX_raw_layer": "filtered UMI counts",
    "Peak_X": "library-size normalized and log1p transformed",
    "Peak_raw_layer": "filtered peak counts",
}

mdata.write_h5mu(OUT)

print(
    f"Written {OUT}: "
    f"{mdata.n_obs} cells, "
    f"{mdata['GEX'].n_vars} genes, "
    f"{mdata['Peak'].n_vars} peaks"
)
