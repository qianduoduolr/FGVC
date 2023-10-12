#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
TASK=$3
CKPT=$4
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NCCL_NET=Socket \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --config $CONFIG --task $TASK --out-indices 1  --checkpoint $CKPT --launcher pytorch ${@:5}  