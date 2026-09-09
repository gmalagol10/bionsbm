#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
from scipy import sparse
from mofapy2.run.entry_point import entry_point


def parse_args():
    p = argparse.ArgumentParser(description="Run repeated MOFA+ fits on a MuData count matrix.")
    p.add_argument("count_matrix", help="Input .h5mu file")
    p.add_argument("path_to_save", help="Output prefix, matching the Mowgli script")
    p.add_argument("start", type=int, help="First run index")
    p.add_argument("--runs", type=int, default=25, help="Exclusive upper run index (default: 25)")
    p.add_argument("--n-latent", type=int, default=None, help="Number of factors; inferred from CellType if omitted")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--convergence", choices=("fast", "medium", "slow"), default="fast")
    p.add_argument("--save-model", action="store_true", help="Also save each fitted MOFA HDF5 model")
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
    raise ValueError("Cannot infer factor count: pass --n-latent or provide a CellType column.")


def common_cells(mdata):
    cells = list(mdata.obs_names)
    common = [c for c in cells if all(c in a.obs_names for a in mdata.mod.values())]
    if not common:
        raise ValueError("No cells are shared by all modalities.")
    if len(common) != len(cells):
        print(f"Using {len(common)} cells shared by every modality (dropped {len(cells)-len(common)}).", flush=True)
    return common


def gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def to_dense_float32(x):
    return (
        x.toarray().astype(np.float32, copy=False)
        if sparse.issparse(x)
        else np.asarray(x, dtype=np.float32)
    )
    
def main():
    args = parse_args()
    path = Path(args.path_to_save)
    folder, name = path.parent, path.name
    folder.mkdir(parents=True, exist_ok=True)

    print("MOFA+ script started at:", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)
    print("CM:", args.count_matrix, "\nPath to save:", args.path_to_save, flush=True)

    mdata = mu.read_h5mu(args.count_matrix)
    cells = common_cells(mdata)
    mods = list(mdata.mod)
    n_latent = args.n_latent or infer_n_latent(mdata)
    arrays = [to_dense_float32(mdata[mod][cells].X) for mod in mods]
    features = [mdata[mod].var_names.astype(str).tolist() for mod in mods]
    model_features = [[f"{mod}:{f}" for f in feat] for mod, feat in zip(mods, features)]
    use_gpu = gpu_available()
    print("Running on:", "cuda" if use_gpu else "cpu", flush=True)
    print("Latent dimension:", n_latent, flush=True)

    for run in range(args.start, args.runs):
        print(time.strftime("%a, %d %b %Y %H:%M:%S"), f"Run {run}/{args.runs}", flush=True)
        ent = entry_point()
        ent.set_data_options(scale_views=True, use_float32=True)
        ent.set_data_matrix(
            [[x.copy()] for x in arrays], likelihoods=["gaussian"] * len(mods),
            views_names=mods, groups_names=["all"], samples_names=[cells],
            features_names=model_features,
        )
        ent.set_model_options(factors=n_latent, spikeslab_weights=True, ard_weights=True)
        ent.set_train_options(
            iter=args.max_iter, convergence_mode=args.convergence,
            dropR2=None, seed=run, gpu_mode=use_gpu,
        )
        ent.build(); ent.run()
        exp = ent.model.getExpectations(only_first_moments=True)
        z, weights = np.asarray(exp["Z"]), exp["W"]

        out = folder / "Runs" / f"Run{run}"
        out.mkdir(parents=True, exist_ok=True)
        dims = [f"Dim_{i}" for i in range(z.shape[1])]
        pd.DataFrame(z.T, index=dims, columns=cells).to_csv(
            out / f"{name}_Embedding.tsv.gz", sep="\t", compression="gzip"
        )
        for mod, feat, w in zip(mods, features, weights):
            pd.DataFrame(np.asarray(w), index=feat, columns=dims).to_csv(
                out / f"{name}_{mod}_Topic.tsv.gz", sep="\t", compression="gzip"
            )
        if args.save_model:
            ent.save(outfile=str(out / f"{name}_Model.hdf5"))


if __name__ == "__main__":
    main()
