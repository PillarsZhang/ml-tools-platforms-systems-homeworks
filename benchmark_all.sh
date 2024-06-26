#!/bin/bash
set -ex

source ~/activate_proxy.sh
export CUDA_VISIBLE_DEVICES=3

for model_name in "facebook/convnext-tiny-224" "microsoft/resnet-50"
do
    for test_item in "torch" "torch_compile" "tvm"
    do
        for batch_size in 1 4 16
        do
            python benchmark.py --model_name $model_name --test_item $test_item --batch_size $batch_size
        done
    done
done