#!/bin/bash

PYTHON=/home/rlange/anaconda3/envs/modularity/bin/python
JOBS=9
CUDA_DEVICES=123

################################
### MNIST BASE - EXPERIMENT 1 ##
################################

L2VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01 0.03162278 0.1)
L1VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01)
DROPVALS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
RUNS=({0..4})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup \
  --save-dir=logs-mnist-base-l2 --device=auto --devices=$CUDA_DEVICES --l2={1} --run={2} \
  ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup \
  --save-dir=logs-mnist-base-l1 --device=auto --devices=$CUDA_DEVICES --l1={1} --run={2} \
  ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup \
  --save-dir=logs-mnist-base-drop --device=auto --devices=$CUDA_DEVICES --drop={1} --run={2} \
  ::: ${DROPVALS[@]} ::: ${RUNS[@]}

###############################
## MNIST WIDE - EXPERIMENT 2 ##
###############################

EXTRA_ARGS="{'channels':(256,256)}"
L2VALS=(0.00001 0.0001 0.001 0.01 0.1)
L1VALS=(0.00001 0.0001 0.001 0.01 0.1)
DROPVALS=(0.1 0.2 0.3 0.4 0.5 0.6)
RUNS=({0..2})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-wide-l2 --device=auto --devices=$CUDA_DEVICES --l2={1} --run={2} \
  ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-wide-l1 --device=auto --devices=$CUDA_DEVICES --l1={1} --run={2} \
  ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-wide-drop --device=auto --devices=$CUDA_DEVICES --drop={1} --run={2} \
  ::: ${DROPVALS[@]} ::: ${RUNS[@]}

###############################
## MNIST DEEP - EXPERIMENT 3 ##
###############################

EXTRA_ARGS="{'channels':(64,64,64,64,64)}"
L2VALS=(0.00001 0.0001 0.001 0.01 0.1)
L1VALS=(0.00001 0.0001 0.001 0.01 0.1)
DROPVALS=(0.1 0.2 0.3 0.4 0.5 0.6)
RUNS=({0..2})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-deep-l2 --device=auto --devices=$CUDA_DEVICES --l2={1} --run={2} \
  ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-deep-l1 --device=auto --devices=$CUDA_DEVICES --l1={1} --run={2} \
  ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=mnist --task=sup --model-args=\"${EXTRA_ARGS}\" \
  --save-dir=logs-mnist-deep-drop --device=auto --devices=$CUDA_DEVICES --drop={1} --run={2} \
  ::: ${DROPVALS[@]} ::: ${RUNS[@]}

##############
### CIFAR10 ##
##############

L2VALS=(0.00001 0.0001 0.001 0.01 0.1)
L1VALS=(0.00001 0.0001 0.001 0.01)
DROPVALS=(0.1 0.2 0.3 0.4 0.5)
RUNS=({0..2})

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=cifar10 --task=sup \
  --save-dir=logs-cifar10-l2 --device=auto --devices=$CUDA_DEVICES --l2={1} --run={2} \
  ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=cifar10 --task=sup \
  --save-dir=logs-cifar10-l1 --device=auto --devices=$CUDA_DEVICES --l1={1} --run={2} \
  ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=$JOBS $PYTHON -m train --dataset=cifar10 --task=sup \
  --save-dir=logs-cifar10-drop --device=auto --devices=$CUDA_DEVICES --drop={1} --run={2} \
  ::: ${DROPVALS[@]} ::: ${RUNS[@]}
