#!/usr/bin/env python3
from pathlib import Path
import re, time
import shutil
import subprocess
import numpy as np
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import snapatac2 as snap
import muon as mu

ROOT = Path("Datasets/GSE268630")
RAW_DIR = ROOT / "raw"
SELECTED_SAMPLES = None  # None uses all 25; otherwise e.g. ["Multiome_10w_FR", "Multiome_10w_NR"]
N_THREADS = 16

MERGED = ROOT / "GSE268630_merged_fragments.tsv.gz"
SAMPLE_MAP = ROOT / "GSE268630_ATAC_sample_map.tsv"
WHITELIST = ROOT / "GSE268630_ATAC_called_barcodes.txt"
GEX_CALLED_BARCODES = ROOT / "GSE268630_GEX_called_barcodes.txt"
PEAK_DIR = ROOT / "peaks"
PEAK_BED = PEAK_DIR / "GSE268630_all_peaks.bed"
SNAP_FRAGMENTS = ROOT / "GSE268630_fragments.snap.h5ad"
RAW_PEAK_MATRIX = ROOT / "GSE268630_ATAC_raw_peak_matrix.h5ad"
OUT = ROOT / "GSE268630_ATAC_processed.h5ad"
TMP = ROOT / "tmp_snapatac2"


def sample_id(path):
	m = re.fullmatch(r"GSM\d+_(.+)_atac_fragments\.tsv\.gz", path.name)
	if m is None:
		raise ValueError(f"Unexpected fragment filename: {path.name}")
	return m.group(1)


def log_quantile(values, q):
	values = np.asarray(values, dtype=float)
	return 10 ** np.quantile(np.log10(values[values > 0]), q)


def merge_fragments(files):
	if MERGED.exists():
		print(f"Using existing {MERGED}", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)
		return

	compressor = ["pigz", "-p", str(N_THREADS), "-c"] if shutil.which("pigz") else ["gzip", "-c"]
	with MERGED.open("wb") as output:
		gzip_proc = subprocess.Popen(compressor, stdin=subprocess.PIPE, stdout=output)
		try:
			for path in files:
				sample = sample_id(path)
				print(f"Merging {sample}", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)
				decompress = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
				awk = subprocess.Popen(
					["awk", "-v", f"suffix=#{sample}",
					 'BEGIN {OFS="\t"} !/^#/ {$4=$4 suffix; print}'],
					stdin=decompress.stdout, stdout=gzip_proc.stdin,
				)
				decompress.stdout.close()
				if awk.wait() or decompress.wait():
					raise RuntimeError(f"Failed while merging {path}")
		finally:
			gzip_proc.stdin.close()
			if gzip_proc.wait():
				raise RuntimeError("Fragment compression failed")


def write_whitelist(samples):
	if not GEX_CALLED_BARCODES.exists():
		raise FileNotFoundError(
			f"{GEX_CALLED_BARCODES} is created by 02_process_GEX.py; run that script first"
		)
	with GEX_CALLED_BARCODES.open() as source, WHITELIST.open("w") as target:
		for line in source:
			barcode = line.strip()
			if barcode.rsplit("#", 1)[-1] in samples:
				target.write(barcode + "\n")


def call_peaks():
	PEAK_DIR.mkdir(parents=True, exist_ok=True)

	prefix = "GSE268630_all_peaks"
	narrow = PEAK_DIR / f"{prefix}_peaks.narrowPeak"
	done = PEAK_DIR / ".macs3_complete"

	macs_complete = (
		done.exists()
		and narrow.exists()
		and narrow.stat().st_size > 0
	)

	if not macs_complete:
		print(
			"Starting MACS3 peak calling",
			time.strftime("%a, %d %b %Y %H:%M:%S"),
			flush=True,
		)

		# Remove potentially incomplete output from a previous interrupted run.
		done.unlink(missing_ok=True)
		PEAK_BED.unlink(missing_ok=True)

		for path in PEAK_DIR.glob(f"{prefix}_*"):
			if path.is_file():
				path.unlink()

		subprocess.run([
			"macs3", "callpeak",
			"-t", str(MERGED),
			"-f", "FRAG",
			"--barcodes", str(WHITELIST),
			"--max-count", "1",
			"-g", "hs",
			"-q", "0.01",
			"-n", prefix,
			"--outdir", str(PEAK_DIR),
		], check=True)

		if not narrow.exists() or narrow.stat().st_size == 0:
			raise RuntimeError(
				f"MACS3 finished without producing a valid file: {narrow}"
			)

		done.touch()

		print(
			f"MACS3 completed: {narrow}",
			time.strftime("%a, %d %b %Y %H:%M:%S"),
			flush=True,
		)

	else:
		print(
			f"Using completed MACS3 output {narrow}",
			time.strftime("%a, %d %b %Y %H:%M:%S"),
			flush=True,
		)

	# Retain only standard GRCh38 chromosomes.
	valid = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
	order = {f"chr{i}": i for i in range(1, 23)} | {
		"chrX": 23,
		"chrY": 24,
	}

	peaks = []
	with narrow.open() as handle:
		for line in handle:
			fields = line.rstrip().split("\t")

			if len(fields) < 3:
				continue

			chrom, start, end = fields[:3]
			start, end = int(start), int(end)

			if chrom in valid and end > start:
				peaks.append((chrom, start, end))

	if not peaks:
		raise RuntimeError(f"No valid peaks found in {narrow}")

	peaks.sort(key=lambda x: (order[x[0]], x[1], x[2]))

	# Merge overlapping MACS3 intervals.
	merged = []
	for chrom, start, end in peaks:
		if merged and merged[-1][0] == chrom and start <= merged[-1][2]:
			merged[-1] = (
				chrom,
				merged[-1][1],
				max(merged[-1][2], end),
			)
		else:
			merged.append((chrom, start, end))

	# Write atomically so an interrupted write cannot leave a partial BED file.
	tmp_bed = PEAK_BED.with_suffix(".bed.tmp")

	with tmp_bed.open("w") as handle:
		for chrom, start, end in merged:
			handle.write(f"{chrom}\t{start}\t{end}\n")

	tmp_bed.replace(PEAK_BED)

	print(
		f"Written {len(merged)} non-overlapping peaks to {PEAK_BED}",
		time.strftime("%a, %d %b %Y %H:%M:%S"),
		flush=True,
	)


def build_peak_matrix():
	TMP.mkdir(exist_ok=True)
	if not SNAP_FRAGMENTS.exists():
		fragments = snap.pp.import_fragments(
			MERGED,
			chrom_sizes=snap.genome.hg38,
			file=SNAP_FRAGMENTS,
			min_num_fragments=1,
			sorted_by_barcode=False,
			whitelist=WHITELIST,
			tempdir=TMP,
			n_jobs=N_THREADS,
		)
		fragments.close()

	if not RAW_PEAK_MATRIX.exists():
		fragments = snap.read(SNAP_FRAGMENTS)
		peaks = snap.pp.make_peak_matrix(
			fragments,
			peak_file=PEAK_BED,
			file=RAW_PEAK_MATRIX,
			chunk_size=500,
			counting_strategy="paired-insertion",
		)
		peaks.close()
		fragments.close()


def process_peak_matrix():
	adata = ad.read_h5ad(RAW_PEAK_MATRIX)
	adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
	adata.obs["barcode"] = adata.obs_names.str.rsplit("#", n=1).str[0]
	adata.obs["sample"] = adata.obs_names.str.rsplit("#", n=1).str[1]

	sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
	min_features = log_quantile(adata.obs["n_genes_by_counts"], 0.05)
	max_features = log_quantile(adata.obs["n_genes_by_counts"], 0.95)
	min_cells = log_quantile(adata.var["n_cells_by_counts"], 0.05)
	max_cells = log_quantile(adata.var["n_cells_by_counts"], 0.95)

	adata = adata[
		adata.obs["n_genes_by_counts"].between(min_features, max_features),
		adata.var["n_cells_by_counts"].between(min_cells, max_cells),
	].copy()
	print("After cell/feature QC:", adata.shape, time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)

	# TSS enrichment and nucleosome signal are intentionally not calculated or filtered.
	adata.layers["raw"] = adata.X.copy()
	sc.pp.normalize_total(adata)
	sc.pp.highly_variable_genes(
		adata, flavor="seurat_v3", subset=False, check_values=False
	)
	cutoff = np.nanquantile(adata.var["variances_norm"], 0.7)
	adata = adata[:, adata.var["variances_norm"] > cutoff].copy()
	sc.pp.log1p(adata)

	cell_mask = np.asarray((adata.X != 0).sum(axis=1)).ravel() > 0
	feature_mask = np.asarray((adata.X != 0).sum(axis=0)).ravel() > 0
	adata = adata[cell_mask, feature_mask].copy()

	# SnapATAC2 uses chr:start-end peak names.
	coords = adata.var_names.str.extract(r"^([^:]+):(\d+)-(\d+)$")
	if not coords.isna().any(axis=None):
		adata.var[["chr", "start", "stop"]] = coords.to_numpy()
		adata.var[["start", "stop"]] = adata.var[["start", "stop"]].astype(int)

	adata.uns["processing"] = {
		"peak_calling": "MACS3 FRAG mode on all selected libraries and Cell Ranger-called barcodes; max-count=1; q=0.01",
		"counting": "SnapATAC2 paired-insertion counts",
		"X": "library-size normalized and log1p transformed",
		"layers/raw": "filtered peak counts",
		"cell_filter": "5th-95th log10 detected-peak quantiles",
		"feature_filter": "5th-95th log10 detected-cell quantiles; upper 30% Seurat-v3 normalized variance",
		"excluded_metrics": "TSS enrichment and nucleosome signal",
	}
	adata.write_h5ad(OUT, compression="gzip")
	print(f"Written {OUT}: {adata.shape}", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)


def main():
	files = sorted(RAW_DIR.rglob("*_atac_fragments.tsv.gz"))
	if SELECTED_SAMPLES is not None:
		files = [x for x in files if sample_id(x) in set(SELECTED_SAMPLES)]
	if not files:
		raise RuntimeError(f"No ATAC fragment files found under {RAW_DIR}")

	samples = [sample_id(x) for x in files]
	if len(samples) != len(set(samples)):
		raise RuntimeError("Duplicate sample identifiers found")
	SAMPLE_MAP.write_text(
		"sample\tfragment_file\n" +
		"".join(f"{sample_id(x)}\t{x}\n" for x in files)
	)
	print(f"Using {len(files)} ATAC libraries", time.strftime("%a, %d %b %Y %H:%M:%S"), flush=True)

	write_whitelist(set(samples))
	merge_fragments(files)
	call_peaks()
	build_peak_matrix()
	process_peak_matrix()

    	# Build a paired MuData object from cells passing both GEX and ATAC QC.
	gex = ad.read_h5ad(ROOT / "GSE268630_GEX_processed.h5ad")
	atac = ad.read_h5ad(OUT)

	common = gex.obs_names[gex.obs_names.isin(atac.obs_names)]
	if len(common) == 0:
		raise RuntimeError("GEX and ATAC have no matching cell barcodes")

	gex = gex[common].copy()
	atac = atac[common].copy()

	mdata = mu.MuData({
		"GEX": gex,
		"Peak": atac,
	})
	mdata.update()

	# Keep useful cell metadata at the global MuData level.
	for column in ["barcode", "sample"]:
		if column in gex.obs:
			mdata.obs[column] = gex.obs.loc[mdata.obs_names, column].values

	mdata.uns["processing"] = {
		"cell_selection": "intersection of cells passing GEX and ATAC modality-specific QC",
		"GEX_X": "library-size normalised and log1p transformed",
		"GEX_raw_layer": "filtered UMI counts",
		"Peak_X": "library-size normalised and log1p transformed",
		"Peak_raw_layer": "filtered peak counts",
	}

	mdata.write_h5mu(ROOT / "GSE268630_GEX_ATAC_processed.h5mu")
	print(
		f"Written MuData: {mdata.n_obs} cells, "
		f"{mdata['GEX'].n_vars} genes, "
		f"{mdata['Peak'].n_vars} peaks",
		time.strftime("%a, %d %b %Y %H:%M:%S"),
		flush=True,
	)


if __name__ == "__main__":
	main()
