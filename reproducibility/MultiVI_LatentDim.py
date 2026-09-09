#!/usr/bin/env python3
import argparse
import gc
import gzip
import time
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import scvi
import torch


LATENT_DIMS = [2, 6, 12, 18, 24]
N_RUNS = 5
COUNT_LAYER = "raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the effect of latent dimensionality in MultiVI."
    )
    parser.add_argument("count_matrix", help="Input RNA+ATAC .h5mu file")
    parser.add_argument("path_to_save", help="Output prefix")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Use the scvi-tools heuristic when omitted",
    )
    parser.add_argument("--rna-mod", default=None)
    parser.add_argument("--atac-mod", default=None)
    return parser.parse_args()


def modality_kind(name, adata):
    text = name.lower().replace("-", "_")
    feature_types = " ".join(
        map(
            str,
            adata.var.get(
                "feature_types",
                pd.Series(dtype=str),
            ).unique(),
        )
    ).lower()

    if (
        any(x in text for x in ("rna", "gex", "gene", "mrna"))
        or "gene expression" in feature_types
    ):
        return "rna"

    if (
        any(x in text for x in ("atac", "peak", "chrom", "access"))
        or "peak" in feature_types
    ):
        return "atac"

    return "unknown"


def choose_modalities(mdata, rna_override=None, atac_override=None):
    kinds = {
        mod: modality_kind(mod, adata)
        for mod, adata in mdata.mod.items()
    }

    rna = rna_override or next(
        (mod for mod, kind in kinds.items() if kind == "rna"),
        None,
    )
    atac = atac_override or next(
        (mod for mod, kind in kinds.items() if kind == "atac"),
        None,
    )

    if rna not in mdata.mod or atac not in mdata.mod:
        raise ValueError(
            "Could not identify one RNA and one ATAC/Peak modality. "
            f"Found: {kinds}. Use --rna-mod and --atac-mod."
        )

    return rna, atac


def common_cells(mdata, modalities):
    cells = list(mdata.obs_names)
    common = [
        cell
        for cell in cells
        if all(cell in mdata[mod].obs_names for mod in modalities)
    ]

    if not common:
        raise ValueError("No cells are shared by RNA and ATAC.")

    if len(common) != len(cells):
        print(
            f"Using {len(common)} paired cells "
            f"(dropped {len(cells) - len(common)} unpaired cells).",
            flush=True,
        )

    return common



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


def run_is_complete(out, name, latent_dim, n_cells):
    embedding = out / f"{name}_Embedding.tsv.gz"
    marker = out / ".complete"

    if (
        marker.is_file()
        and embedding.is_file()
        and embedding.stat().st_size > 0
    ):
        return True

    if valid_gzip_tsv(embedding, latent_dim, n_cells):
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
        "MultiVI latent-dimension script started at:",
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

    original = mu.read_h5mu(args.count_matrix)
    rna, atac = choose_modalities(
        original,
        rna_override=args.rna_mod,
        atac_override=args.atac_mod,
    )
    cells = common_cells(original, (rna, atac))

    mdata = mu.MuData(
        {
            rna: original[rna][cells].copy(),
            atac: original[atac][cells].copy(),
        }
    )

    for mod in (rna, atac):
        if COUNT_LAYER not in mdata[mod].layers:
            raise KeyError(
                f"Modality {mod!r} has no {COUNT_LAYER!r} layer. "
                f"Available layers: {list(mdata[mod].layers.keys())}"
            )

    print(f"Selected {rna!r} + {atac!r}", flush=True)

    scvi.model.MULTIVI.setup_mudata(
        mdata,
        rna_layer=COUNT_LAYER,
        atac_layer=COUNT_LAYER,
        modalities={
            "rna_layer": rna,
            "atac_layer": atac,
        },
    )

    train_kwargs = {
        "early_stopping": True,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
    }
    if args.max_epochs is not None:
        train_kwargs["max_epochs"] = args.max_epochs

    pending = []
    for latent_dim in LATENT_DIMS:
        for run in range(N_RUNS):
            out = folder / "LatentDim" / f"LT_{latent_dim}" / f"Run{run}"
            if run_is_complete(out, name, latent_dim, len(cells)):
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

        scvi.settings.seed = run
        np.random.seed(run)
        torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run)

        model = scvi.model.MULTIVI(
            mdata,
            n_genes=mdata[rna].n_vars,
            n_regions=mdata[atac].n_vars,
            n_latent=latent_dim,
        )
        model.train(**train_kwargs)

        latent = model.get_latent_representation()
        out = folder / "LatentDim" / f"LT_{latent_dim}" / f"Run{run}"
        out.mkdir(parents=True, exist_ok=True)
        (out / ".complete").unlink(missing_ok=True)

        atomic_to_csv(
            pd.DataFrame(
                latent.T,
                index=[f"Dim_{i}" for i in range(latent.shape[1])],
                columns=mdata.obs_names,
            ),
            out / f"{name}_Embedding.tsv.gz",
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
