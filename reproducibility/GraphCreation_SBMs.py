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
import bionsbm
from nsbm import nsbm

DATA = "Datasets/GSE268630/GSE268630_Peak_GEX_processed.h5mu"
REQUESTED_SIZES = [100, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
TABLE = Path("Tables/Time_bionSBM_vs_nSBM_cells.tsv")


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


def add_differences(df):
	df["dt_bionsbm_vs_nsbm"] = df["t_bionsbm"] - df["t_nsbm"]
	df["dram_bionsbm_vs_nsbm"] = df["ram_bionsbm"] - df["ram_nsbm"]
	df["dt_pct_vs_nsbm"] = 100 * df["dt_bionsbm_vs_nsbm"] / df["t_nsbm"]
	df["dram_pct_vs_nsbm"] = 100 * df["dram_bionsbm_vs_nsbm"] / df["ram_nsbm"]

	return df


def load_results_graph():
	raw = [
		"n_cells",
		"t_bionsbm",
		"ram_bionsbm",
		"t_nsbm",
		"ram_nsbm",
	]

	if not TABLE.exists():
		return pd.DataFrame(columns=raw)

	df = pd.read_csv(TABLE, sep="\t")

	if not set(raw).issubset(df.columns):
		raise ValueError(f"{TABLE} must contain {raw}")

	df[raw] = df[raw].apply(pd.to_numeric, errors="coerce")
	df = df.dropna(subset=raw)
	df = df[np.isfinite(df[raw]).all(axis=1)]
	df["n_cells"] = df["n_cells"].astype(int)

	return add_differences(
		df.drop_duplicates("n_cells", keep="last").sort_values("n_cells")
	)


def save_result_graph(df, row):
	df = pd.concat(
		[df[df["n_cells"] != row["n_cells"]], pd.DataFrame([row])],
		ignore_index=True,
	)

	df["n_cells"] = df["n_cells"].astype(int)
	df = add_differences(df.sort_values("n_cells"))

	TABLE.parent.mkdir(parents=True, exist_ok=True)
	tmp = TABLE.with_name(TABLE.name + ".tmp")
	df.to_csv(tmp, sep="\t", index=False)
	tmp.replace(TABLE)

	return df


def run_bionsbm(obj):
	bionsbm.model.bionsbm(obj=obj)


def run_nsbm(obj):
	model = nsbm()
	model.make_graph_multiple_df(
		obj["Peak"].to_df().T,
		[obj[key].to_df().T for key in obj.mod if key != "Peak"],
	)


def main():
	mdata = mu.read_h5mu(DATA)
	sizes = available_sizes(mdata, REQUESTED_SIZES)

	results = load_results_graph()
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

		cell_file = tmp_dir / f"graph_{n_cells}_cells.npy"
		bio_result = tmp_dir / f"graph_bionsbm_{n_cells}.json"
		nsbm_result = tmp_dir / f"graph_nsbm_{n_cells}.json"

		try:
			for path in (cell_file, bio_result, nsbm_result):
				path.unlink(missing_ok=True)

			save_selected_cells(mdata_sub, cell_file)
			del mdata_sub
			gc.collect()

			t_bionsbm, ram_bionsbm = run_fresh_worker(
				cell_file,
				bio_result,
				mode="bionsbm",
			)

			t_nsbm, ram_nsbm = run_fresh_worker(
				cell_file,
				nsbm_result,
				mode="nsbm",
			)

			row = {
				"n_cells": n_cells,
				"t_bionsbm": t_bionsbm,
				"ram_bionsbm": ram_bionsbm / 1e9,
				"t_nsbm": t_nsbm,
				"ram_nsbm": ram_nsbm / 1e9,
			}

			results = save_result_graph(results, row)
			completed.add(n_cells)

			dt = t_bionsbm - t_nsbm
			dram = ram_bionsbm / 1e9 - ram_nsbm / 1e9
			dt_pct = 100 * dt / t_nsbm
			dram_pct = 100 * dram / (ram_nsbm / 1e9)

			print(
				f"[{n_cells:>9} cells] "
				f"| bionSBM: {t_bionsbm:.5f}s, {ram_bionsbm / 1e9:.5f}GB "
				f"| nSBM: {t_nsbm:.5f}s, {ram_nsbm / 1e9:.5f}GB "
				f"| Δtime: {dt:+.5f}s ({dt_pct:+.5f}%) "
				f"| ΔRAM: {dram:+.5f}GB ({dram_pct:+.5f}%)",
				flush=True,
			)
			print(f"Saved: {TABLE}", flush=True)

		except Exception:
			print(
				f"Failed at {n_cells} cells. "
				f"Completed sizes remain saved in {TABLE}.",
				flush=True,
			)
			raise

		finally:
			for path in (cell_file, bio_result, nsbm_result):
				path.unlink(missing_ok=True)
			gc.collect()


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
		obj = load_selected_cells(sys.argv[2])

		if sys.argv[4] == "bionsbm":
			write_worker_result(sys.argv[3], run_bionsbm, obj)
		elif sys.argv[4] == "nsbm":
			write_worker_result(sys.argv[3], run_nsbm, obj)
		else:
			raise ValueError(sys.argv[4])
	else:
		main()
