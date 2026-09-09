#!/usr/bin/env python3
import argparse
import random
import time
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData


def parse_args():
    p = argparse.ArgumentParser(description="Run repeated COBOLT fits on a MuData count matrix.")
    p.add_argument("count_matrix", help="Input .h5mu file")
    p.add_argument("path_to_save", help="Output prefix, matching the Mowgli script")
    p.add_argument("start", type=int, help="First run index")
    p.add_argument("--runs", type=int, default=25, help="Exclusive upper run index (default: 25)")
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


def common_cells(mdata):
    cells = list(mdata.obs_names)
    common = [c for c in cells if all(c in a.obs_names for a in mdata.mod.values())]
    if not common:
        raise ValueError("No cells are shared by all modalities.")
    if len(common) != len(cells):
        print(f"Using {len(common)} cells shared by every modality (dropped {len(cells)-len(common)}).", flush=True)
    return common


def as_csr(x):
    return x.tocsr().astype(np.float32) if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32))


def main():
    args = parse_args()
    path = Path(args.path_to_save)
    folder, name = path.parent, path.name
    folder.mkdir(parents=True, exist_ok=True)

    print("COBOLT script started at:", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)
    print("Running on:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("CM:", args.count_matrix, "\nPath to save:", args.path_to_save, flush=True)

    mdata = mu.read_h5mu(args.count_matrix)
    cells = common_cells(mdata)
    n_latent = infer_n_latent(mdata)

    single = []
    for mod, adata in mdata.mod.items():
        adata = adata[cells]
        single.append(SingleData(
            feature_name=str(mod), dataset_name="paired",
            feature=adata.var_names.astype(str).to_numpy(),
            count=as_csr(adata.X), barcode=np.asarray(cells, dtype=str),
        ))
    dataset = MultiomicDataset.from_singledata(*single)
    print(dataset, flush=True)
    print("Latent dimension:", n_latent, flush=True)

    for run in range(args.start, args.runs):
        print(time.strftime("%a, %d %b %Y %H:%M:%S"), f"Run {run}/{args.runs}", flush=True)
        random.seed(run); np.random.seed(run); torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run)

        model = Cobolt(dataset=dataset, n_latent=n_latent, lr=0.0005)
        model.train()
        latent, barcodes = model.get_all_latent(correction=True)
        barcodes = [str(x).split("~", 1)[-1] for x in barcodes]

        out = folder / "Runs" / f"Run{run}"
        out.mkdir(parents=True, exist_ok=True)
        dims = [f"Dim_{i}" for i in range(latent.shape[1])]
        pd.DataFrame(latent.T, index=dims, columns=barcodes).to_csv(
            out / f"{name}_Embedding.tsv.gz", sep="\t", compression="gzip"
        )

        for mod, beta in zip(dataset.omic, model.model.get_beta()):
            pd.DataFrame(beta, index=dataset.dt[mod]["feature"], columns=dims).to_csv(
                out / f"{name}_{mod}_Topic.tsv.gz", sep="\t", compression="gzip"
            )


if __name__ == "__main__":
    main()
