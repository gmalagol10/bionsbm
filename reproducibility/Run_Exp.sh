#!/bin/bash
set -euo pipefail

CONDA_SH=/store24/project24/ladcol_012/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR=/store24/project24/ladcol_012/bionSBM_revision

tool=$1
shift


case "$tool" in
    Mowgli_Resources|Mowgli|Mowgli_LatentDim)
        CONDA_ENV=mowgli
        script="$SCRIPT_DIR/${tool}.py"
        ;;
    *)
        CONDA_ENV=tm
        script="$SCRIPT_DIR/${tool}.py"
        ;;
esac

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export MPLBACKEND=Agg


echo "Environment: $CONDA_ENV"
echo "Python: $(command -v python)"
echo "Script: $script"
echo "Arguments: $*"

python "$script" "$@"

echo "Script $script completed successfully"
