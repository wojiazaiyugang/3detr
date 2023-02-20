#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--dataset_name tooth \
--nqueries 256 \
--test_ckpt outputs/6/checkpoint_best.pth \
--test_only \
--enc_type masked
