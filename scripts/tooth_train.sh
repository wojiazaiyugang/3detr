#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--dataset_name tooth \
--max_epoch 1080 \
--enc_type masked \
--enc_dropout 0.3 \
--nqueries 256 \
--base_lr 5e-4 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/6 \
--batchsize_per_gpu 6 \
--dataset_root_dir /media/3TB/data/xiaoliutech/tooth_det_3detr_farthest_sample \
--meta_data_dir /media/3TB/data/xiaoliutech/tooth_det_3detr_farthest_sample
