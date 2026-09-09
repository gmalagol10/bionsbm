import warnings
warnings.filterwarnings("ignore")
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
from scipy import sparse

from helps import *
import shutil
from SHARE_topic import SHARE_topic

DATA = "Datasets/GSE268630/GSE268630_Peak_GEX_processed.h5mu"
REQUESTED_SIZES = [1000, 5000, 10000, 50000, 100000]
TABLE = Path("Tables/Time_ShareTopic_cells.tsv")


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
			adata = source[mod]

			missing = pd.Index(cells).difference(adata.obs_names)
			if len(missing):
				raise RuntimeError(
					f"{mod}: {len(missing)} selected cells are missing; "
					f"examples: {list(missing[:5])}"
				)

			mods[mod] = adata[cells].to_memory()
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


def run_sharetopic(obj):
	tmp = Path("TempShare")

	# SHARE-Topic explicitly expects ATAC first and RNA second.
	atac = obj["Peak"].copy()
	rna = obj["GEX"].copy()

	try:
		atac.X = atac.layers["raw"].copy()
		rna.X = rna.layers["raw"].copy()

		if not sparse.issparse(atac.X):
			atac.X = sparse.csr_matrix(atac.X)
		if not sparse.issparse(rna.X):
			rna.X = sparse.csr_matrix(rna.X)

		# SHARE-Topic assumes active ATAC-region indices are contiguous 0..R-1.
		# After cell subsampling many peaks can be all-zero; remove them so the
		# remaining columns are reindexed before SHARE-Topic sees the matrix.
		keep = np.asarray(atac.X.getnnz(axis=0)).ravel() > 0
		n_removed = int((~keep).sum())
		atac = atac[:, keep].copy()

		if np.any(np.asarray(atac.X.getnnz(axis=1)).ravel() == 0):
			raise RuntimeError("At least one selected cell has zero ATAC counts")

		print(
			f"SHARE-Topic input: {atac.n_obs} cells, "
			f"{atac.n_vars} active peaks, {rna.n_vars} genes "
			f"({n_removed} zero peaks removed)",
			flush=True,
		)

		gamma = 1
		tau = 0.5
		n_topics = int(rna.obs["sample"].dropna().nunique())
		n_samples = 100
		n_burnin = 1
		batch_size = 64
		alpha = 50 / n_topics
		beta = 0.1

		model = SHARE_topic(
			atac,
			rna,
			n_topics,
			alpha,
			beta,
			gamma,
			tau,
		)

		theta, lam, phi = model.fit(
			batch_size,
			n_samples,
			n_burnin,
			dev="cpu",
			save_data=False,
		)
		# SHARE-Topic's WAIC() hard-codes the ATAC batch size to 100.
		# Call the two components explicitly so both use the safe batch size.
		model.WAIC_ATAC(theta, phi, "cpu", batch_size=batch_size)
		model.WAIC_RNA(batch_size, theta, lam, "cpu")

		m_theta = theta[n_samples - 1:].mean(axis=0)
		m_theta /= m_theta.sum(axis=1)[:, None]

		m_phi = phi[n_samples - 1:].mean(axis=0)
		m_phi /= m_phi.sum(axis=1)[:, None]

		m_lam = lam[n_samples - 1:].mean(axis=0)
		m_lam /= m_lam.sum(axis=1)[:, None]

		top_cell = pd.DataFrame(
			m_theta.cpu().detach().numpy(),
			index=rna.obs_names,
			columns=[f"Topic_{i}" for i in range(n_topics)],
		).T

		top_peak = pd.DataFrame(
			m_phi.cpu().detach().numpy(),
			index=[f"Topic_{i}" for i in range(n_topics)],
			columns=atac.var_names,
		)

		top_gex = pd.DataFrame(
			m_lam.cpu().detach().numpy(),
			index=[f"Topic_{i}" for i in range(n_topics)],
			columns=rna.var_names,
		)

		tmp.mkdir(parents=True, exist_ok=True)
		top_cell.to_csv(
			tmp / "Topic_Cell.tsv.gz",
			sep="\t",
			compression="gzip",
		)
		top_peak.to_csv(
			tmp / "Topic_Peak.tsv.gz",
			sep="\t",
			compression="gzip",
		)
		top_gex.to_csv(
			tmp / "Topic_GEX.tsv.gz",
			sep="\t",
			compression="gzip",
		)

	finally:
		shutil.rmtree(tmp, ignore_errors=True)


def main():
	mdata = mu.read_h5mu(DATA)
	sizes = available_sizes(mdata, REQUESTED_SIZES)

	results = load_results(TABLE, "t_sharetopic", "ram_sharetopic")
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

		cell_file = tmp_dir / f"sharetopic_{n_cells}_cells.npy"
		result_file = tmp_dir / f"sharetopic_{n_cells}.json"

		try:
			cell_file.unlink(missing_ok=True)
			result_file.unlink(missing_ok=True)

			save_selected_cells(mdata_sub, cell_file)
			del mdata_sub
			gc.collect()

			elapsed, ram = run_fresh_worker(cell_file, result_file)

			row = {
				"n_cells": n_cells,
				"t_sharetopic": elapsed,
				"ram_sharetopic": ram / 1e9,
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
		write_worker_result(sys.argv[3], run_sharetopic, obj)
	else:
		main()
