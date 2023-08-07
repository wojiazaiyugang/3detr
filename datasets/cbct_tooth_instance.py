# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import pickle

import numpy as np
from copy import deepcopy
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
from utils import angle_between
from .scan_tooth import to_line_set

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "/media/3TB/data/xiaoliutech/cbct_tooth_point_cloud_det_3detr_20230726+20230727+20230728+20230729+20230730+20230731+20230732+20230733"  ## Replace with path to dataset
DATASET_METADATA_DIR = "/media/3TB/data/xiaoliutech/cbct_tooth_point_cloud_det_3detr_20230726+20230727+20230728+20230729+20230730+20230731+20230732+20230733"  ## Replace with path to dataset


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 32 + 1  # 32个牙 一个背景
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "background": 0,
        }
        for i in range(1, 5):
            for j in range(1, 9):
                self.type2class[f"tooth_{i}{j}"] = len(self.type2class)
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # Semantic Segmentation Classes. Not used in 3DETR
        self.num_class_semseg = 32 + 1  # 32个牙 一个背景
        self.type2class_semseg = {
            "tooth": 0,
        }
        for i in range(1, 5):
            for j in range(1, 9):
                self.type2class_semseg[f"tooth_{i}{j}"] = len(self.type2class_semseg)
        self.class2type_semseg = {
            self.type2class_semseg[t]: t for t in self.type2class_semseg
        }
        self.nyu40ids_semseg = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32]
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
            self,
            center,
            heading_class,
            heading_residual,
            size_class,
            size_residual,
            box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class ScannetDetectionDataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            root_dir=None,
            meta_data_dir=None,
            num_points=40000,
            use_color=False,
            use_height=False,
            augment=False,
            use_random_cuboid=False,
            random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        assert split_set in ["train", "val"]
        self.split_set = split_set
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:13]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(meta_data_dir, f"scannetv2_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [
                sname for sname in self.scan_names if sname in all_scan_names
            ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_vert.npy")
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )
        semantic_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        )
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_bbox.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        if self.augment and False:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, [instance_labels, semantic_labels]
            )
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
        # #
        point_cloud, choices = pc_util.random_sampling(
           point_cloud, 50000, return_choices=True
        )
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        for _c in self.dataset_config.nyu40ids_semseg:
            sem_seg_labels[
                semantic_labels == _c
                ] = self.dataset_config.nyu40id2class_semseg[_c]

        pcl_color = pcl_color[choices]

        target_bboxes_mask[0: instance_bboxes.shape[0]] = 1
        target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]


        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            pass

            # if np.random.random() > 0.5:
            #     # Flipping along the YZ plane
            #     point_cloud[:, 0] = -1 * point_cloud[:, 0]
            #     target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
            #
            # if np.random.random() > 0.5:
            #     # Flipping along the XZ plane
            #     point_cloud[:, 1] = -1 * point_cloud[:, 1]
            #     target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            angle = 12  # -5 ~ +5 degree
            rot_angle_x = (np.random.random() * np.pi / (angle / 2)) - np.pi / angle  # -5 ~ +5 degree
            rot_mat_x = pc_util.rotx(rot_angle_x)
            rot_angle_y = (np.random.random() * np.pi / (angle / 2)) - np.pi / angle  # -5 ~ +5 degree
            rot_mat_y = pc_util.roty(rot_angle_y)
            rot_angle_z = (np.random.random() * np.pi / (angle / 2)) - np.pi / angle  # -5 ~ +5 degree
            rot_mat_z = pc_util.rotz(rot_angle_z)
            rot_mat = np.dot(rot_mat_x, np.dot(rot_mat_y, rot_mat_z))

            # show = False


            # if show:
            #     import open3d as o3d
            #     old_pc = o3d.geometry.PointCloud()
            #     old_pc.points = o3d.utility.Vector3dVector(deepcopy(point_cloud[:, 0:3]))
            #     red = np.array([0.93, 0.93, 0.93])
            #     old_pc.colors = o3d.utility.Vector3dVector(np.array([red for _ in range(point_cloud.shape[0])]))
            #     line_sets = to_line_set(target_bboxes)
            #     o3d.visualization.draw_geometries([old_pc] + line_sets)

            def compute_bbox(point_cloud, semantic_labels):
                target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
                bboxes = []
                for label in np.unique(semantic_labels):
                    if label == 0:
                        continue
                    pc = point_cloud[semantic_labels == label]
                    bbox_start, bbox_end = np.min(pc, axis=0), np.max(pc, axis=0)
                    box_center, box_size = (bbox_start + bbox_end) / 2, bbox_end - bbox_start
                    bboxes.append(np.concatenate([box_center, box_size]))
                bboxes = np.array(bboxes)
                target_bboxes[:bboxes.shape[0]] = bboxes
                return target_bboxes

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = compute_bbox(point_cloud, semantic_labels)

            # if show:
            #
            #     new_pc = o3d.geometry.PointCloud()
            #     new_pc.points = o3d.utility.Vector3dVector(deepcopy(point_cloud[:, 0:3]))
            #     red = np.array([0.93, 0.93, 0.93])
            #     new_pc.colors = o3d.utility.Vector3dVector(np.array([red for _ in range(point_cloud.shape[0])]))
            #     line_sets = to_line_set(target_bboxes)
            #     o3d.visualization.draw_geometries([new_pc] + line_sets)

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        # target_axisfls = target_axisfls.astype(np.float32)[:, 0:3]
        # target_axisfls_normalized = shift_scale_points(
        #     target_axisfls[None, ...],
        #     src_range=[
        #         point_cloud_dims_min[None, ...],
        #         point_cloud_dims_max[None, ...],
        #     ],
        #     dst_range=self.center_normalizing_range,
        #
        # )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        # target_axisfls_normalized = target_axisfls_normalized.squeeze(0)
        # target_axisfls_normalized = target_axisfls_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(np.float32)
        # ret_dict["gt_axisfls_normalized"] = target_axisfls_normalized.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0: instance_bboxes.shape[0]] = [
            self.dataset_config.nyu40id2class[int(x)]
            for x in instance_bboxes[:, -1][0: instance_bboxes.shape[0]]
        ]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict
