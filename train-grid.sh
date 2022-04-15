#!/bin/bash

PYTHON=/home/rlange/anaconda3/envs/modularity/bin/python
JOBS=9

######################
## MNIST SUPERVISED ##
######################

CUDA_DEVICES=123 # cuda:0 on dolores is different hardware, which doesn't guarantee deterministic results!
DATASETS=(mnist)
TASKS=(sup)
L2VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01 0.03162278 0.1)
L1VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01 0.03162278 0.1)
DROPVALS=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)
RUNS=({0..8})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset={1} --task={2} --save-dir=logs-{2}-l2 --device=auto --devices=$CUDA_DEVICES --l2={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset={1} --task={2} --save-dir=logs-{2}-l1 --device=auto --devices=$CUDA_DEVICES --l1={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset={1} --task={2} --save-dir=logs-{2}-drop --device=auto --devices=$CUDA_DEVICES --drop={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${DROPVALS[@]} ::: ${RUNS[@]}

########################
## CIFAR10 SUPERVISED ##
########################

CUDA_DEVICES=123 # cuda:0 on dolores is different hardware, which doesn't guarantee deterministic results!
DATASETS=(cifar10)
TASKS=(sup)
L2VALS=(0.00001 0.0001 0.001 0.01 0.1)
L1VALS=(0.00001 0.0001 0.001 0.01)
RUNS=({0..2})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset={1} --task={2} --save-dir=logs-{1}-{2}-l2 --device=auto --devices=$CUDA_DEVICES --l2={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset={1} --task={2} --save-dir=logs-{1}-{2}-l1 --device=auto --devices=$CUDA_DEVICES --l1={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L1VALS[@]} ::: ${RUNS[@]}
