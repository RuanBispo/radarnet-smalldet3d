#!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
PORT=${PORT:-28501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=1 \
python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port=$PORT \
  ./tools/train.py ./projects/configs/redformer/redformer.py \
  --launcher pytorch ${@:3} \
  --deterministic \
  --resume-from ./work_dirs/redformer/epoch_7.pth \
  --no-validate
  # > terminal_output_dist_train.txt
  # --no-validate \

# torchrun python -m torch.distributed.run
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#   --nproc_per_node 2 \
#   --master_port=$PORT \
#   $(dirname "$0")/train.py ./projects/configs/redformer/redformer.py \
#   --launcher pytorch ${@:3} \
#   --deterministic \
#   --no-validate

# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#   $(dirname "$0")/train.py  $CONFIG --launcher pytorch ${@:3} --deterministic \
#   --no-validate
  # --resume-from ./work_dirs/redformer/latest.pth \
