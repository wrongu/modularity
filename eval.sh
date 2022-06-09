#!/bin/bash

# Activate conda, and the 'modularity' env within it
source ~/anaconda3/etc/profile.d/conda.sh
conda activate modularity

echo "Using" `which python`

CUDA_DEVICES=(1 2 3)
JOBS=12

####################################
## BASIC STATS ON ALL CHECKPOINTS ##
####################################

# This first pass over all .ckpt files computes stats like val_loss among others
FILES=($(find $1* -name "*.ckpt" | grep -v "best.ckpt" | grep -v "dummy.ckpt" | sort))
echo "=== BASIC STATS ON ALL ${#FILES[@]} CHECKPOINTS ==="
parallel --linebuffer --tag --link --jobs=$JOBS \
    python -m eval --ckpt-file={1} --device=cuda:{2} --skip-modularity \
    ::: ${FILES[@]} ::: ${CUDA_DEVICES[@]}

##########################################
## IDENTIFY 'BEST' CHECKPOINT PER MODEL ##
##########################################

# Create a 'best.ckpt' alias for whichever checkpoint has lowest validation loss, per model
echo "=== IDENTIFY BEST CHECKPOINT PER RUN ==="
CKPT_DIRS=($(find $1* -type d | grep weights | grep -v dummy | sort))
parallel --linebuffer --jobs=$JOBS \
    python -m create_best_ckpt {1} --recurse --field=val_loss --mode=min \
    ::: ${CKPT_DIRS[@]}

##########################################
## FANCY STATS ON BEST CHECKPOINTS ONLY ##
##########################################

# Do modularity metrics only on the "best" checkpoint per model
FILES=($(find $1* -name "best.ckpt" | sort))
echo "=== RUN MODULARITY METRICS ON ${#FILES[@]} CHECKPOINTS ==="
MODMETRICS="forward_cov,backward_hess,forward_jac,backward_jac,forward_cov_norm,backward_hess_norm,forward_jac_norm,backward_jac_norm"
parallel --linebuffer --tag --link --jobs=$JOBS python -m eval --ckpt-file={1} --device=cuda:{2} \
  --modularity-metrics=$MODMETRICS ::: ${FILES[@]} ::: ${CUDA_DEVICES[@]}

###################################################
## STATS ON DUMMY CHECKPOINTS (UNTRAINED MODELS) ##
###################################################

FILES=($(find $1* -name "dummy.ckpt" | sort | head -n 100))
echo "=== RUN MODULARITY METRICS ON ${#FILES[@]} DUMMY CHECKPOINTS ==="
parallel --linebuffer --tag --link --jobs=$JOBS python -m eval --ckpt-file={1} --device=cuda:{2} \
  --modularity-metrics=$MODMETRICS ::: ${FILES[@]} ::: ${CUDA_DEVICES[@]}
