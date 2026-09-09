#!/usr/bin/env python3
import argparse
import gc
import gzip
import random
import time
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import softmax
from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData


LATENT_DIMS = [2, 6, 12, 18, 24]
N_RUNS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the effect of COBOLT latent dimensionality."
    )
    parser.add_argument("count_matrix", help="Input .h5mu file")
    parser.add_argument(
        "path_to_save",
        help="Output prefix, for example Datasets/PBMC/COBOLT/STLD",
    )
    return parser.parse_args()


def common_cells(mdata):
    cells = list(mdata.obs_names)
    common = [
        cell
        for cell in cells
        if all(cell in adata.obs_names for adata in mdata.mod.values())
    ]

    if not common:
        raise ValueError("No cells are shared by all modalities.")

    if len(common) != len(cells):
        print(
            f"Using {len(common)} cells shared by every modality "
            f"(dropped {len(cells) - len(common)}).",
            flush=True,
        )

    return common


def as_csr(x):
    if sparse.issparse(x):
        return x.tocsr().astype(np.float32)

    return sparse.csr_matrix(
        np.asarray(x, dtype=np.float32)
    )



def atomic_to_csv(frame, path):
    """Write a gzip TSV atomically."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    frame.to_csv(tmp, sep="\t", compression="gzip")
    tmp.replace(path)


def valid_gzip_tsv(path, expected_rows, expected_values):
    """Validate the full gzip stream and table dimensions."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rt") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            if len(header) != expected_values + 1:
                return False
            rows = sum(1 for _ in handle)
        return rows == expected_rows
    except (OSError, EOFError, UnicodeDecodeError):
        return False


def write_complete_marker(out):
    tmp = out / ".complete.tmp"
    marker = out / ".complete"
    tmp.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    tmp.replace(marker)


def run_is_complete(out, name, latent_dim, n_cells, dataset):
    marker = out / ".complete"
    expected = [
        out / f"{name}_Embedding.tsv.gz",
        out / f"{name}_Topic_Cell.tsv.gz",
        out / f"{name}_MaxTopic.tsv.gz",
    ] + [
        out / f"{name}_{mod}_Topic.tsv.gz"
        for mod in dataset.omic
    ]

    if marker.is_file() and all(
        path.is_file() and path.stat().st_size > 0
        for path in expected
    ):
        return True

    checks = [
        valid_gzip_tsv(expected[0], latent_dim, n_cells),
        valid_gzip_tsv(expected[1], latent_dim, n_cells),
        valid_gzip_tsv(expected[2], n_cells, 1),
    ]
    checks.extend(
        valid_gzip_tsv(
            out / f"{name}_{mod}_Topic.tsv.gz",
            len(dataset.dt[mod]["feature"]),
            latent_dim,
        )
        for mod in dataset.omic
    )

    if all(checks):
        out.mkdir(parents=True, exist_ok=True)
        write_complete_marker(out)
        return True
    return False

def main():
    args = parse_args()

    path = Path(args.path_to_save)
    folder = path.parent
    name = path.name
    folder.mkdir(parents=True, exist_ok=True)

    print(
        "COBOLT latent-dimension script started at:",
        time.strftime("%a, %d %b %Y %H:%M:%S"),
        flush=True,
    )
    print(
        "Running on:",
        "cuda" if torch.cuda.is_available() else "cpu",
        flush=True,
    )
    print("CM:", args.count_matrix, flush=True)
    print("Path to save:", args.path_to_save, flush=True)

    mdata = mu.read_h5mu(args.count_matrix)
    cells = common_cells(mdata)

    # Match the working COBOLT.py exactly: construct COBOLT inputs from adata.X.
    single = []
    for mod, adata in mdata.mod.items():
        adata = adata[cells]

        single.append(
            SingleData(
                feature_name=str(mod),
                dataset_name="paired",
                feature=adata.var_names.astype(str).to_numpy(),
                count=as_csr(adata.X),
                barcode=np.asarray(cells, dtype=str),
            )
        )

    dataset = MultiomicDataset.from_singledata(*single)
    print(dataset, flush=True)

    pending = []
    for latent_dim in LATENT_DIMS:
        for run in range(N_RUNS):
            out = folder / "LatentDim" / f"LT_{latent_dim}" / f"Run{run}"
            if run_is_complete(
                out, name, latent_dim, len(cells), dataset
            ):
                print(
                    f"Latent dim {latent_dim} Run {run}: complete, skipping",
                    flush=True,
                )
            else:
                pending.append((latent_dim, run))

    if not pending:
        print("All latent-dimension runs are complete.", flush=True)
        return

    print(
        f"Resuming from latent dim {pending[0][0]}, run {pending[0][1]}",
        flush=True,
    )

    for latent_dim, run in pending:
        print(
            time.strftime("%a, %d %b %Y %H:%M:%S"),
            f"Latent dim {latent_dim} Run {run}/{N_RUNS}",
            flush=True,
        )

        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run)

        model = Cobolt(
            dataset=dataset,
            n_latent=latent_dim,
            lr=0.0005,
        )
        model.train()

        latent, barcodes = model.get_all_latent(correction=True)
        barcodes = [
            str(barcode).split("~", 1)[-1]
            for barcode in barcodes
        ]
        dims = [f"Dim_{i}" for i in range(latent.shape[1])]

        out = folder / "LatentDim" / f"LT_{latent_dim}" / f"Run{run}"
        out.mkdir(parents=True, exist_ok=True)
        (out / ".complete").unlink(missing_ok=True)

        embedding = pd.DataFrame(
            latent.T,
            index=dims,
            columns=barcodes,
        )
        atomic_to_csv(
            embedding,
            out / f"{name}_Embedding.tsv.gz",
        )

        topic_cell = pd.DataFrame(
            softmax(latent, axis=1).T,
            index=dims,
            columns=barcodes,
        )
        atomic_to_csv(
            topic_cell,
            out / f"{name}_Topic_Cell.tsv.gz",
        )

        atomic_to_csv(
            pd.DataFrame(
                topic_cell.to_numpy().argmax(axis=0),
                index=barcodes,
                columns=["cluster"],
            ),
            out / f"{name}_MaxTopic.tsv.gz",
        )

        for mod, beta in zip(
            dataset.omic,
            model.model.get_beta(),
        ):
            atomic_to_csv(
                pd.DataFrame(
                    beta,
                    index=dataset.dt[mod]["feature"],
                    columns=dims,
                ),
                out / f"{name}_{mod}_Topic.tsv.gz",
            )

        write_complete_marker(out)
        print(
            f"Latent dim {latent_dim} Run {run}: saved and marked complete",
            flush=True,
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
