#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
contamination=$3
pca=$4

mkdir $xp_dir;

# GTSRB training   sh scripts/gtsrb_if.sh gtsrb_0_if 0 0.1 0
python baseline_isoForest.py --xp_dir $xp_dir --dataset gtsrb --n_estimators 100 --max_samples 256 \
    --contamination $contamination --out_frac 0 --seed $seed --unit_norm_used l1 --gcn 0 --pca $pca;
