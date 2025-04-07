#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

PY="python experiments_pps_random_split_by_features.py"

conda activate general

cd src

for method in MLPE CC PACC EMQ KDEy-ML ; do
  $PY --method $method &
done



