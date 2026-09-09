#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import scvi
import torch
from scipy import sparse


def parse_args():
    p = argparse.ArgumentParser(description="Run MultiVI for RNA+ATAC or totalVI for RNA+ADT MuData.")
    p.add_argument("count_matrix", help="Input .h5mu file")
    p.add_argument("path_to_save", help="Output prefix, matching the Mowgli script")
    p.add_argument("start", type=int, help="First run index")
    p.add_argument("--runs", type=int, default=25, help="Exclusive upper run index (default: 25)")
    p.add_argument("--n-latent", type=int, default=None, help="Latent dimension; inferred from CellType if omitted")
    p.add_argument("--max-epochs", type=int, default=None, help="Use scvi-tools heuristic if omitted")
    p.add_argument("--rna-mod", default=None)
    p.add_argument("--other-mod", default=None, help="ATAC or ADT modality; auto-detected if omitted")
    p.add_argument("--batch-key", default=None, help="Optional batch column in the RNA modality")
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def infer_n_latent(mdata):
    candidates = []
    if "CellType" in mdata.obs:
        candidates.append(mdata.obs["CellType"])
    candidates += [mdata.obs[c] for c in mdata.obs if c.endswith(":CellType")]
    candidates += [a.obs["CellType"] for a in mdata.mod.values() if "CellType" in a.obs]
    for x in candidates:
        n = x.dropna().nunique()
        if n > 1:
            return int(n)
    raise ValueError("Cannot infer latent dimension: pass --n-latent or provide a CellType column.")


def modality_kind(name, adata):
    s = name.lower().replace("-", "_")
    values = " ".join(map(str, adata.var.get("feature_types", pd.Series(dtype=str)).unique())).lower()
    if any(x in s for x in ("rna", "gex", "gene", "mrna")) or "gene expression" in values:
        return "rna"
    if any(x in s for x in ("atac", "peak", "chrom", "access")) or "peak" in values:
        return "atac"
    if any(x in s for x in ("adt", "protein", "prot", "antibody", "cite")) or "antibody capture" in values:
        return "protein"
    return "unknown"


def choose_modalities(mdata, rna_override, other_override):
    kinds = {m: modality_kind(m, a) for m, a in mdata.mod.items()}
    rna = rna_override or next((m for m, k in kinds.items() if k == "rna"), None)
    other = other_override or next((m for m, k in kinds.items() if k in {"atac", "protein"}), None)
    if rna not in mdata.mod or other not in mdata.mod:
        raise ValueError(f"Could not identify RNA and ATAC/ADT modalities. Found: {kinds}; use --rna-mod and --other-mod.")
    kind = modality_kind(other, mdata[other])
    if kind not in {"atac", "protein"}:
        raise ValueError(f"Modality {other!r} is not recognisable as ATAC or ADT/protein.")
    return rna, other, kind


def common_cells(mdata, mods):
    cells = list(mdata.obs_names)
    common = [c for c in cells if all(c in mdata[m].obs_names for m in mods)]
    if not common:
        raise ValueError("No cells are shared by the selected modalities.")
    if len(common) != len(cells):
        print(f"Using {len(common)} paired cells (dropped {len(cells)-len(common)} unpaired cells).", flush=True)
    return common


def main():
    args = parse_args()
    path = Path(args.path_to_save)
    folder, name = path.parent, path.name
    folder.mkdir(parents=True, exist_ok=True)

    print("scvi-tools script started at:", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)
    print("Running on:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("CM:", args.count_matrix, "\nPath to save:", args.path_to_save, flush=True)

    original = mu.read_h5mu(args.count_matrix)
    rna, other, kind = choose_modalities(original, args.rna_mod, args.other_mod)
    cells = common_cells(original, (rna, other))
    mdata = mu.MuData({rna: original[rna][cells].copy(), other: original[other][cells].copy()})
    n_latent = args.n_latent or infer_n_latent(original)
    print(f"Selected {rna!r} + {other!r}; model: {'MultiVI' if kind == 'atac' else 'totalVI'}", flush=True)
    print("Latent dimension:", n_latent, flush=True)

    setup = {"modalities": {"rna_layer": rna}}
    count_layer = "raw"

    if kind == "atac":
        scvi.model.MULTIVI.setup_mudata(
            mdata,
            rna_layer=count_layer,
            atac_layer=count_layer,
            modalities={
                "rna_layer": rna,
                "atac_layer": other,
            },
        )
    else:
        for mod in (rna, other):
            x = mdata[mod].layers[count_layer]
            if sparse.issparse(x):
                mdata[mod].layers[count_layer] = x.toarray()

        scvi.model.TOTALVI.setup_mudata(
            mdata,
            rna_layer=count_layer,
            protein_layer=count_layer,
            modalities={
                "rna_layer": rna,
                "protein_layer": other,
            },
        )
        
        
    for run in range(args.start, args.runs):
        print(time.strftime("%a, %d %b %Y %H:%M:%S"), f"Run {run}/{args.runs}", flush=True)
        scvi.settings.seed = run
        np.random.seed(run); torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run)

        if kind == "atac":
            model = scvi.model.MULTIVI(
                mdata, n_genes=mdata[rna].n_vars,
                n_regions=mdata[other].n_vars, n_latent=n_latent,
            )
        else:
            model = scvi.model.TOTALVI(mdata, n_latent=n_latent)

        train_kwargs = {"early_stopping": True, "accelerator": "gpu" if torch.cuda.is_available() else "cpu", "devices": 1}
        if args.max_epochs is not None:
            train_kwargs["max_epochs"] = args.max_epochs
        model.train(**train_kwargs)
        latent = model.get_latent_representation()

        out = folder / "Runs" / f"Run{run}"
        out.mkdir(parents=True, exist_ok=True)
        dims = [f"Dim_{i}" for i in range(latent.shape[1])]
        pd.DataFrame(latent.T, index=dims, columns=mdata.obs_names).to_csv(
            out / f"{name}_Embedding.tsv.gz", sep="\t", compression="gzip"
        )
        if args.save_model:
            model.save(out / f"{name}_Model", overwrite=True, save_anndata=False)


if __name__ == "__main__":
    main()
