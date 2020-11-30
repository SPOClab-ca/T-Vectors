#!/bin/bash

K=5
POOL_MAX=4
mkdir predictions -p

function all_pool_values {
  DS_NAME="${1%%.*}"
  TARGET=$2

  for (( i = 1; i < $((POOL_MAX+1)); i++ )); do
      python3 analysis.py "extracted_vectors/$1" --knn $K --save-predictions "predictions/${DS_NAME}_pool_$i.npz" \
              --pool-vectors $i $TARGET
  done
}

function all_targets() {
    DS=$1
    all_pool_values $DS
    all_pool_values $DS "--predict-session"
    all_pool_values $DS "--predict-dataset"
}

all_targets bci_iv_2a.npz
all_pool_values "bci_iv_2a.npz" "--predict-events"

# Events by themselves because MMI dataset is different in this case
all_targets mmi_1020.npz
all_pool_values "mmi_1020_lrh.npz" "--predict-events"
