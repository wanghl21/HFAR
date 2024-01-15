#!/bin/bash

parameters_method=("KRED")
parameters_alpha=("0.13")
for method in "${parameters_method[@]}"
do
    for alpha in "${parameters_alpha[@]}"
    do
    echo "parameters: $method" "$alpha"
    python qua_rec_classifier_contrastive_learning.py "$method" "$alpha"
    done
done
echo "done！"
