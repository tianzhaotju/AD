#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
pca=$3

mkdir $xp_dir;

# GTSRB training   sh scripts/gtsrb_kde.sh gtsrb_0_kde 0 0
python baseline_kde.py --xp_dir $xp_dir --dataset gtsrb --kernel gaussian --seed $seed --GridSearchCV 1 \
    --out_frac 0 --unit_norm_used l1 --pca $pca --gcn 0;
