#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# 数据集里面的数据集路径也要改，命令行里的好像不会生效，没仔细看，都改了就完事了

python main.py \
--dataset_name cbct_tooth_instance \
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
--checkpoint_dir outputs/cbct_tooth_instance_6 \
--batchsize_per_gpu 8 \
--dataset_root_dir /media/3TB/data/xiaoliutech/cbct_tooth_det_3detr_2X_farthest \
--meta_data_dir /media/3TB/data/xiaoliutech/cbct_tooth_det_3detr_2X_farthest
