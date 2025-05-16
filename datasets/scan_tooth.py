import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import open3d as o3d
import numpy as np
import torch
from algorithm_assistant import TriangleMesh, ToothKeypoints, ToothAxis, TOOTH, Tooth, ArcType, Sphere, Point3D, Color
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import TypeAlias

from config import use_axis_head, use_kps_head, KEY_POINT_NAMES
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_normalize import get_normalize_transformation
from utils.pc_util import scale_points, shift_scale_points

Data: TypeAlias = Tuple[TriangleMesh, Dict[Tooth, Tuple[ToothAxis, ToothKeypoints]]] # 数据，牙齿网格和牙齿信息

class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 32 + 1  # 32个牙 一个背景
        self.num_angle_bin = 1
        self.max_num_obj = 1

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
        self.split_set = split_set
        self.datas: List[Data] = []

        for dataset_name in ["20230228", "20230229", "20230230", "20230411", "20231214"]:
            dataset = Path("/media/8TB/dataset").joinpath(dataset_name)
            if split_set == "train":
                data_names = dataset.joinpath(f"train.txt").read_text().splitlines() + dataset.joinpath(f"val.txt").read_text().splitlines()
            elif split_set == "val":
                data_names = dataset.joinpath(f"test.txt").read_text().splitlines()
            else:
                raise NotImplementedError

            for data_name in tqdm(data_names, desc=f"加载数据集{dataset_name} {split_set}"):
                data_dir = dataset.joinpath(data_name)
                data_dict = {}
                if dataset_name in ["20230228", "20230229", "20230230", "20230411"]:
                    keypoint_file = data_dir.joinpath("为了迁移牙轴做的检测结果.json")
                    if not keypoint_file.exists():
                        print(f"关键点文件{keypoint_file=}不存在")
                        continue
                    for detect_result in json.loads(keypoint_file.read_text()):
                        tooth = TOOTH.get_tooth_by_category(category=detect_result["category"])
                        tooth_axis = ToothAxis.from_dict(detect_result["data"]["axis"])
                        try:
                            tooth_keypoints = tooth.keypoints_type.from_dict(detect_result["data"]["keypoints"])
                        except KeyError as err:
                            print(f"{data_dir=} {err=}")
                            continue
                        data_dict[tooth] = tooth_axis, tooth_keypoints
                else:
                    keypoint_file = data_dir.joinpath("keypoint.json")
                    if not keypoint_file.exists():
                        print(f"关键点文件{keypoint_file=}不存在")
                        continue
                    for tid_str, tooth_data in json.loads(keypoint_file.read_text()).items():
                        tooth = TOOTH.get_tooth_by_tid(tid=int(tid_str))
                        tooth_axis = ToothAxis.from_dict(tooth_data)
                        tooth_keypoints = tooth.keypoints_type.from_dict(tooth_data)
                        data_dict[tooth] = tooth_axis, tooth_keypoints
                # 把1区和4区的关键点md方向对调，跟md轴保持一致，减小训练难度
                tooth_keypoints: ToothKeypoints
                tooth: Tooth
                for tooth, (_, tooth_keypoints) in data_dict.items():
                    if tooth.is_area2_tooth or tooth.is_area3_tooth:
                        continue
                    if hasattr(tooth_keypoints, "ldc") and hasattr(tooth_keypoints, "lmc"):
                        tooth_keypoints.ldc, tooth_keypoints.lmc = tooth_keypoints.lmc, tooth_keypoints.ldc
                    if hasattr(tooth_keypoints, "occm") and hasattr(tooth_keypoints, "occd"):
                        tooth_keypoints.occm, tooth_keypoints.occd = tooth_keypoints.occd, tooth_keypoints.occm
                    if hasattr(tooth_keypoints, "bdc") and hasattr(tooth_keypoints, "bmc"):
                        tooth_keypoints.bmc, tooth_keypoints.bdc = tooth_keypoints.bdc, tooth_keypoints.bmc
                    if hasattr(tooth_keypoints, "mrm") and hasattr(tooth_keypoints, "mrd"):
                        tooth_keypoints.mrd, tooth_keypoints.mrm = tooth_keypoints.mrm, tooth_keypoints.mrd
                    if hasattr(tooth_keypoints, "fcd") and hasattr(tooth_keypoints, "fcm"):
                        tooth_keypoints.fcm, tooth_keypoints.fcd = tooth_keypoints.fcd, tooth_keypoints.fcm
                    if hasattr(tooth_keypoints, "nfcd") and hasattr(tooth_keypoints, "nfcm"):
                        tooth_keypoints.nfcm, tooth_keypoints.nfcd = tooth_keypoints.nfcd, tooth_keypoints.nfcm
                for arc_type in [ArcType.UPPER, ArcType.LOWER]:
                    mesh_file = data_dir.joinpath(f"{arc_type.lower()}_jaw.ply")
                    if not mesh_file.exists():
                        print(f"网格文件{mesh_file=}不存在")
                        continue
                    mesh = TriangleMesh.from_file(mesh_file)
                    mesh_data = {tooth: tooth_data for tooth, tooth_data in data_dict.items() if tooth.arc_type == arc_type}
                    if len(mesh_data) == 0:
                        print(f"没有牙齿数据{data_dir=} {arc_type=}")
                        continue
                    self.datas.append((mesh, mesh_data))
                # if len(self.datas) > 1:
                #     break
        print(f"数据加载完成 {split_set=} {len(self.datas)=}")
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx: int):
        mesh, tooth_data = self.datas[idx] # 注意深拷贝问题
        # 随机选个牙
        tooth_colors = np.unique(mesh.colors[(mesh.colors[:, 0] != 0.8) | (mesh.colors[:, 1] != 0.8) | (mesh.colors[:, 2] != 0.8)], axis=0)
        for tooth_color in np.random.permutation(tooth_colors):
            tooth: Tooth = TOOTH.get_tooth_by_color(color=Color(red=int(tooth_color[0] * 255), green=int(tooth_color[1] * 255), blue=int(tooth_color[2] * 255)))
            # tooth = TOOTH.tooth_23
            if tooth in tooth_data:
                tooth_axis, tooth_keypoints = tooth_data[tooth]
                break
        else:
            print(f"{idx=}没有检测框")
            return self.__getitem__(0)
        # 获取这个牙的所有点
        tooth_vertices = mesh.vertices[(mesh.colors == tooth.color.to_rgb_float_tuple()).all(axis=1)]
        # 随机选一个点作为点击点
        click_point = Point3D(*random.choice(tooth_vertices))
        # 裁剪
        new_mesh = mesh.crop_by_sphere(sphere=Sphere(center=click_point, radius=15))
        assert new_mesh, f"{new_mesh=}是空"
        # 平移到原点
        transformation = get_normalize_transformation(mesh=new_mesh, click_point=click_point)
        new_mesh = new_mesh.transform(transformation)
        tooth_axis = tooth_axis.transform(transformation)
        tooth_keypoints = tooth_keypoints.transform(transformation)

        if self.split_set == "train": # 数据增强
            transformation = np.identity(4)
            # 三个轴随机旋转
            matrix = o3d.geometry.get_rotation_matrix_from_xyz(
                (
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                )
            )
            transformation[:3, :3] = matrix
            new_mesh = new_mesh.transform(transformation)
            tooth_axis = tooth_axis.transform(transformation)
            tooth_keypoints = tooth_keypoints.transform(transformation)

        # from algorithm_assistant import visualizer
        # visualizer.add_points([click_point])
        # visualizer.add_tooth_axis(axis=tooth_axis, point=tooth_keypoints.occc)
        # visualizer.add_tooth_keypoints(tooth=tooth, tooth_keypoints=tooth_keypoints)
        # visualizer.add_triangle_mesh(triangle_mesh=mesh)
        # visualizer.add_triangle_mesh(triangle_mesh=new_mesh)
        # visualizer.show()

        sample_index = np.random.choice(len(new_mesh.vertices), 10000)
        point_cloud, colors = new_mesh.vertices[sample_index], new_mesh.colors[sample_index]

        # point_cloud = PointCloud(points=vertices)
        # visualizer.add_point_cloud(point_cloud, radius=0.15 * transformation[0,0])
        # visualizer.show()

        tooth_point_cloud = point_cloud[(colors == tooth.color.to_rgb_float_tuple()).all(axis=1)]
        bbox_start, bbox_end = np.min(tooth_point_cloud, axis=0), np.max(tooth_point_cloud, axis=0)
        box_center, box_size = (bbox_start + bbox_end) / 2, bbox_end - bbox_start
        instance_bboxes = np.array([np.concatenate([box_center, box_size, np.array([tooth.category])])])

        if use_axis_head or use_kps_head:
            if use_axis_head:
                axisfl = np.array([tooth_axis.axisfl.to_numpy()])
                axismd = np.array([tooth_axis.axismd.to_numpy()])
                axisie = np.array([tooth_axis.axisie.to_numpy()])

            if use_kps_head:
                key_points = {}
                for kp in KEY_POINT_NAMES:
                    point: Point3D = getattr(tooth_keypoints, kp, Point3D(0,0,0))
                    key_points[kp] = np.array([point.to_numpy()], dtype=np.float32)

        pcl_color = np.array([0])
        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        if use_axis_head:
            target_axisfls = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
            target_axismds = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
            target_axisies = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
            target_axisfls[0: axisfl.shape[0], :] = axisfl[:, 0:3]
            target_axismds[0: axismd.shape[0], :] = axismd[:, 0:3]
            target_axisies[0: axisie.shape[0], :] = axisie[:, 0:3]

        if use_kps_head:
            for kp in KEY_POINT_NAMES:
                k = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
                k[0: key_points[kp].shape[0], :] = key_points[kp][:, 0:3]
                key_points[kp] = k

        target_bboxes_mask[0: instance_bboxes.shape[0]] = 1
        target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = (point_cloud.min(axis=0)[:3]).astype(np.float32)
        point_cloud_dims_max = (point_cloud.max(axis=0)[:3]).astype(np.float32)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        if use_kps_head:
            keypoints_normalized = {}
            for kp in key_points:
                key_points[kp] = key_points[kp].astype(np.float32)
                keypoints_normalized[kp] = shift_scale_points(
                    key_points[kp][None, ...],
                    src_range=[
                        point_cloud_dims_min[None, ...],
                        point_cloud_dims_max[None, ...],
                    ],
                    dst_range=self.center_normalizing_range,
                )
                keypoints_normalized[kp] = keypoints_normalized[kp].squeeze(0)
                keypoints_normalized[kp] = keypoints_normalized[kp] * target_bboxes_mask[..., None]

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
        if use_axis_head:
            ret_dict["gt_axisfls"] = target_axisfls.astype(np.float32)
            ret_dict["gt_axismds"] = target_axismds.astype(np.float32)
            ret_dict["gt_axisies"] = target_axisies.astype(np.float32)
        if use_kps_head:
            ret_dict["gt_keypoints"] = key_points
            ret_dict["gt_keypoints_normalizeds"] = keypoints_normalized
        ret_dict["tid"] = tooth.tid
        return ret_dict
