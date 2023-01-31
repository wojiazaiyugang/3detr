#!/bin/bash

python main.py \
--dataset_name scannet \
--nqueries 256 \
--test_ckpt checkpoints/scannet_masked_ep1080.pth \
--test_only \
--enc_type masked \
--dataset_root_dir /home/yujiannan/Projects/3detr/scannet/scannet_train_detection_data \
--meta_data_dir /media/3TB/data/ScanNet \
--batchsize_per_gpu 1 \
--dataset_num_workers 0