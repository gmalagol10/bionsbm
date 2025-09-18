#!/bin/bash

CONDA_ENV=topmod
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

python /store24/project24/ladcol_012/TopicModelling/${1}.py $2 $3 $4 $5 $6

echo "Script ${exp}.py completed"
