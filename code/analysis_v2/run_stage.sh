#!/bin/bash
# usage: v2/run_stage.sh <script> [args...]   (one BLAS thread per worker; parallelism over permutations / folds)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
s=$1; shift; python3 v2/$s "$@" > v2/results/${s%.py}.log 2>&1
