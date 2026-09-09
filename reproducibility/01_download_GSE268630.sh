#!/usr/bin/env bash
set -euo pipefail

OUT="Datasets/GSE268630"
URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE268nnn/GSE268630/suppl"

mkdir -p "$OUT" "$OUT/raw"

command -v wget >/dev/null || { echo "wget is required" >&2; exit 1; }
command -v tar  >/dev/null || { echo "tar is required" >&2; exit 1; }

# Download the 25 joint filtered feature-barcode H5 files and the RAW tar archive.
wget -c -r -np -nd -e robots=off \
	--accept='GSE268630_*_filtered_feature_bc_matrix.h5,GSE268630_RAW.tar' \
	-P "$OUT" "$URL/"

n_h5=$(find "$OUT" -maxdepth 1 -name 'GSE268630_*_filtered_feature_bc_matrix.h5' | wc -l)
[[ "$n_h5" -eq 25 ]] || {
	echo "Expected 25 H5 matrices, found $n_h5. Re-run the script before continuing." >&2
	exit 1
}
[[ -s "$OUT/GSE268630_RAW.tar" ]] || {
	echo "Missing $OUT/GSE268630_RAW.tar" >&2
	exit 1
}

# The tar contains the sample-level ATAC fragment files and their indexes.
if [[ ! -f "$OUT/raw/.extracted" ]]; then
	tar -xf "$OUT/GSE268630_RAW.tar" -C "$OUT/raw"
	touch "$OUT/raw/.extracted"
fi

n_frag=$(find "$OUT/raw" -type f -name '*_atac_fragments.tsv.gz' | wc -l)
echo "Downloaded $n_h5 H5 matrices and extracted $n_frag fragment files into $OUT"
