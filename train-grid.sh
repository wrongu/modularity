#!/bin/bash

PYTHON=/home/rlange/.venv/mytorch/bin/python
ENV="CUBLAS_WORKSPACE_CONFIG=:4096:8"

CUDA_DEVICES=123 # cuda:0 on dolores is different hardware, which doesn't guarantee deterministic results!
DATASETS=(mnist)
TASKS=(sup unsup)
L2VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01 0.03162278 0.1)
L1VALS=(0.00001 0.00003162278 0.0001 0.0003162278 0.001 0.003162278 0.01 0.03162278 0.1)
DROPVALS=(0.05 0.15 0.25 0.35 0.45 0.55)
RUNS=({0..5})

parallel --linebuffer --tag --jobs=30 $ENV $PYTHON -m train --dataset={1} --task={2} --save-dir=logs --device=auto --devices=$CUDA_DEVICES --l2={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L2VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=30 $ENV $PYTHON -m train --dataset={1} --task={2} --save-dir=logs --device=auto --devices=$CUDA_DEVICES --l1={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${L1VALS[@]} ::: ${RUNS[@]}

parallel --linebuffer --tag --jobs=30 $ENV $PYTHON -m train --dataset={1} --task={2} --save-dir=logs --device=auto --devices=$CUDA_DEVICES --drop={3} --run={4} \
  ::: ${DATASETS[@]} ::: ${TASKS[@]} ::: ${DROPVALS[@]} ::: ${RUNS[@]}
