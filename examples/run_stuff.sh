#!/bin/bash

model=${1}
tasks=${2}
log_file=${3}
use_pytorch=${4:-1}
cls_batch_size=${5:-32}

MKL_THREADING_LAYER=GNU python ${model}.py --tasks ${tasks} --log_file ${log_file} --use_pytorch ${use_pytorch} --cls_batch_size ${cls_batch_size}
