import warnings
warnings.filterwarnings("ignore")
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd

from helps import *
import shutil
import mowgli

DATA = "Datasets/GSE268630/GSE268630_Peak_GEX_processed.h5mu"
REQUESTED_SIZES = [1000, 5000, 10000, 50000, 100000]
TABLE = Path("Tables/Time_Mowgli_cells.tsv")


def load_results(path, time_col, ram_col):
	cols = ["n_cells", time_col, ram_col]
	if not path.exists():
		return pd.DataFrame(columns=cols)

	df = pd.read_csv(path, sep="\t")
	if not set(cols).issubset(df.columns):
		raise ValueError(f"{path} must contain {cols}")

	df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
	df = df[np.isfinite(df[time_col]) & np.isfinite(df[ram_col])]
	df["n_cells"] = df["n_cells"].astype(int)

	return df.drop_duplicates("n_cells", keep="last").sort_values("n_cells")


def save_result(df, row, path):
	df = pd.concat(
		[df[df["n_cells"] != row["n_cells"]], pd.DataFrame([row])],
		ignore_index=True,
	)
	df["n_cells"] = df["n_cells"].astype(int)
	df = df.sort_values("n_cells")

	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(path.name + ".tmp")
	df.to_csv(tmp, sep="\t", index=False)
	tmp.replace(path)

	return df


def available_sizes(mdata, requested):
	requested = np.asarray(requested, dtype=int)
	skipped = requested[requested > mdata.n_obs]

	if len(skipped):
		print(
			f"Skipping unavailable sizes {skipped.tolist()}; "
			f"object has {mdata.n_obs} cells",
			flush=True,
		)

	return requested[requested <= mdata.n_obs]


def save_selected_cells(mdata_sub, path):
	np.save(path, np.asarray(mdata_sub.obs_names, dtype="U"), allow_pickle=False)


def load_selected_cells(path):
	cells = np.load(path, allow_pickle=False).astype(str)
	source = mu.read_h5mu(DATA, backed=True)
	mods = {}

	try:
		for mod in source.mod:
			backed = source[mod]

			missing = pd.Index(cells).difference(backed.obs_names)
			if len(missing):
				raise RuntimeError(
					f"{mod}: {len(missing)} selected cells are missing; "
					f"examples: {list(missing[:5])}"
				)

			tmp = backed[cells].to_memory()

			layers = {
				key: value.copy()
				for key, value in tmp.layers.items()
				if isinstance(key, str) and key
			}

			mods[mod] = ad.AnnData(
				X=tmp.X.copy(),
				obs=tmp.obs.copy(),
				var=tmp.var.copy(),
				layers=layers,
			)

			del tmp

	finally:
		try:
			source.file.close()
		except Exception:
			pass
		del source

	obj = mu.MuData(mods)
	gc.collect()
	return obj


def run_fresh_worker(cell_file, result_file, mode=None):
	cmd = [
		sys.executable,
		str(Path(__file__).resolve()),
		"--worker",
		str(cell_file),
		str(result_file),
	]

	if mode is not None:
		cmd.append(mode)

	subprocess.run(cmd, check=True)

	with open(result_file) as handle:
		result = json.load(handle)

	if "error" in result:
		raise RuntimeError(result["error"])

	return float(result["time"]), int(result["ram"])


def write_worker_result(result_file, fn, obj):
	try:
		_, elapsed, ram = run_with_peak_increase(
			fn,
			obj,
			poll_s=0.005,
			discard_result=True,
		)
		result = {"time": elapsed, "ram": ram}
	except BaseException:
		result = {"error": traceback.format_exc()}

	with open(result_file, "w") as handle:
		json.dump(result, handle)

	if "error" in result:
		raise RuntimeError(result["error"])


def run_mowgli(obj):
	tmp = Path("TEMPMowgli")

	try:
		adatas = {}

		for mod in obj.mod:
			adata = obj[mod].copy()
			adata.X = adata.X.toarray()
			adatas[mod] = adata

		obj = mu.MuData(adatas)

		n_latent = int(obj["GEX"].obs["sample"].dropna().nunique())
		h_regularization = {mod: 5e-2 for mod in obj.mod}

		model = mowgli.models.MowgliModel(
			latent_dim=n_latent,
			h_regularization=h_regularization,
		)
		model.train(obj, device="cpu")

		tmp.mkdir(parents=True, exist_ok=True)
		pd.DataFrame(
			obj.obsm["W_OT"].T,
			columns=obj.obs_names,
			index=[f"Dim_{i}" for i in range(n_latent)],
		).to_csv(tmp / "Embedding.tsv.gz", sep="	", compression="gzip")

	finally:
		shutil.rmtree(tmp, ignore_errors=True)


def main():
	mdata = mu.read_h5mu(DATA)
	sizes = available_sizes(mdata, REQUESTED_SIZES)

	results = load_results(TABLE, "t_mowgli", "ram_mowgli")
	completed = set(results["n_cells"].astype(int))
	print(f"Completed sizes: {sorted(completed)}", flush=True)

	tmp_dir = Path("TempResourceBenchmark")
	tmp_dir.mkdir(parents=True, exist_ok=True)

	for n_cells in sizes:
		n_cells = int(n_cells)

		if n_cells in completed:
			print(f"Skipping completed size {n_cells}", flush=True)
			continue

		print(f"Starting with {n_cells}", flush=True)

		mdata_sub = muon_subsample(
			mdata,
			{"cells": n_cells},
			strata_column="sample",
		)

		cell_file = tmp_dir / f"mowgli_{n_cells}_cells.npy"
		result_file = tmp_dir / f"mowgli_{n_cells}.json"

		try:
			cell_file.unlink(missing_ok=True)
			result_file.unlink(missing_ok=True)

			save_selected_cells(mdata_sub, cell_file)
			del mdata_sub
			gc.collect()

			elapsed, ram = run_fresh_worker(cell_file, result_file)

			row = {
				"n_cells": n_cells,
				"t_mowgli": elapsed,
				"ram_mowgli": ram / 1e9,
			}

			results = save_result(results, row, TABLE)
			completed.add(n_cells)

			print(
				f"[{n_cells:>9} cells] | "
				f"time: {elapsed:.5f}s | RAM: {ram / 1e9:.5f}GB",
				flush=True,
			)
			print(f"Saved: {TABLE}", flush=True)

		except Exception:
			print(
				f"Failed at {n_cells} cells; completed TSV rows were preserved",
				flush=True,
			)
			raise

		finally:
			cell_file.unlink(missing_ok=True)
			result_file.unlink(missing_ok=True)
			gc.collect()


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
		obj = load_selected_cells(sys.argv[2])
		write_worker_result(sys.argv[3], run_mowgli, obj)
	else:
		main()
