#!/usr/bin/env bash

python3 train_cifar.py --dataset cifar10 --model resnet --depth 20 --norm RNT --norm_power 1.0 --gpu 1 --checkpoint path_to_model --train-batch 128
python3 test_cifar.py --dataset cifar10 --model resnet --depth 20 --norm RNT --gpu 2 --epsilons 0.02 --resume /path_to_saved_checkpoint --output_file_path file_name.csv --test-batch 128
