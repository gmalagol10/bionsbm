#!/bin/bash

#Update the path and the env
CONDA_ENV=topmod
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

echo $1
echo $2
echo $3
echo $4
#Update the path
python /store24/project24/ladcol_012/bionSBM/${1}.py $2 $3 $4 $5 $6

echo "Script ${exp}.py completed"
