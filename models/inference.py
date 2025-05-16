from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from algorithm_assistant import TriangleMesh, ToothDetectResult3D, BBox3D, Point3D, TOOTH, ToothAxis, Sphere

from datasets.scan_tooth import ScannetDatasetConfig
from main import make_args_parser
from models.model_3detr import build_3detr, Model3DETR
from utils.ap_calculator import get_ap_config_dict, parse_predictions, flip_axis_to_depth
from utils.pc_normalize import get_normalize_transformation

model: Optional[Model3DETR] = None
device = torch.device("cuda")
dataset_config = ScannetDatasetConfig()

def init_model() -> None:
    """
    初始化模型
    :return:
    """
    global model
    parser = make_args_parser()
    args, _ = parser.parse_known_args()
    model, _ = build_3detr(args, dataset_config)
    # model_file = Path("/home/yujiannan/Projects/XiaoLiuInfer/models/scan_tooth_det_with_axis_and_kps_3detr_20230228-new-axis+20230229-new-axis+20230230-new-axis+20230411-new-axis+20231214_mAP0.25_96.07_mAP0.5_95.49_mAP0.75_91.38_20240218.pth")
    model_file = Path("/home/yujiannan/Projects/3detr/outputs/单牙点击检测/11/checkpoint_best.pth")
    model.load_state_dict(torch.load(str(model_file), map_location=torch.device("cpu"))["model"], strict=False)
    model.to(device)
    model.eval()

def parse_output(point_clouds, outputs, config_dict, centroid, m) -> List[ToothDetectResult3D]:
    batch_pred_map_cls = parse_predictions(outputs["box_corners"],
                                           outputs["sem_cls_prob"],
                                           outputs["objectness_prob"],
                                           point_clouds,
                                           config_dict,
                                           outputs)

    bboxes, detect_results = [], []
    for pred in batch_pred_map_cls[0]:
        cls_prob, box, obj_score, axisfl, axismd, axisie, keypoints = pred
        cls = int(np.argmax(cls_prob))
        score = obj_score * cls_prob[cls]  # detect_result返回的置信度，是score和cls_prob的乘积
        # cls_prob按照概率从大到小排序，组成一个list，每个元素是一个tuple，tuple的第一个元素是概率，第二个元素是类别
        category_score = {int(category): float(prob) for category, prob in enumerate(cls_prob)}
        box = flip_axis_to_depth(box)
        box = box * m + centroid
        for kp in keypoints:
            keypoints[kp] = keypoints[kp] * m + centroid
            keypoints[kp] = {
                "x": keypoints[kp][0],
                "y": keypoints[kp][1],
                "z": keypoints[kp][2],
            }
        # axisfl = flip_axis_to_depth(axisfl)
        # axisfl = axisfl * m + centroid
        # 把轴向处理成单位向量
        axisfl = axisfl / np.linalg.norm(axisfl)
        axismd = axismd / np.linalg.norm(axismd)
        axisie = axisie / np.linalg.norm(axisie)
        axisfl = axisfl.tolist()
        axismd = axismd.tolist()
        axisie = axisie.tolist()
        start, end = np.min(box, axis=0), np.max(box, axis=0)
        xyzxyz = tuple(map(float, tuple(np.concatenate([start, end]))))
        bbox3d = BBox3D(point1=Point3D(xyzxyz[0], xyzxyz[1], xyzxyz[2]), point7=Point3D(xyzxyz[3], xyzxyz[4], xyzxyz[5]))
        category = cls
        tooth = TOOTH.get_tooth_by_category(category=category)
        detect_results.append(ToothDetectResult3D(bbox=bbox3d,
                                             category=category,
                                             label=tooth.name,
                                             score=float(score), # detect_result返回的置信度，是obj_score * cls_score
                                             tooth_keypoints=tooth.keypoints_type.from_dict(keypoints),
                                                  tooth_axis=ToothAxis.from_dict({
                                                     "axisfl": {
                                                        "x": axisfl[0],
                                                        "y": axisfl[1],
                                                        "z": axisfl[2],
                                                     },
                                                     "axismd": {
                                                            "x": axismd[0],
                                                            "y": axismd[1],
                                                            "z": axismd[2],

                                                     },
                                                     "axisie": {
                                                            "x": axisie[0],
                                                            "y": axisie[1],
                                                            "z": axisie[2],
                                                     },
                                                 }),
                                             # data={
                                             #     "axis": {
                                             #         "axisfl": {
                                             #            "x": axisfl[0],
                                             #            "y": axisfl[1],
                                             #            "z": axisfl[2],
                                             #         },
                                             #         "axismd": {
                                             #                "x": axismd[0],
                                             #                "y": axismd[1],
                                             #                "z": axismd[2],
                                             #
                                             #         },
                                             #         "axisie": {
                                             #                "x": axisie[0],
                                             #                "y": axisie[1],
                                             #                "z": axisie[2],
                                             #         },
                                             #     },
                                             #     "keypoints": keypoints,
                                             #     "category_score": category_score,
                                             #     "obj_score": float(obj_score)
                                             # }
                                                  ))
    return detect_results

def infer(mesh: TriangleMesh, click_point: Point3D, tid: int) -> List[ToothDetectResult3D]:
    """
    牙齿检测
    :param mesh:
    :param click_point:
    :param tid:
    :return:
    """
    global model
    if model is None:
        init_model()

    new_mesh = mesh.crop_by_sphere(sphere=Sphere(center=click_point, radius=15))
    assert new_mesh is not None, f"点击点{click_point}不在网格内"
    transformation = get_normalize_transformation(mesh=new_mesh, click_point=click_point)
    new_mesh = new_mesh.transform(transformation)
    sample_index = np.random.choice(len(new_mesh.vertices), 10000)
    point_cloud = new_mesh.vertices[sample_index]
    point_clouds = torch.from_numpy(point_cloud).unsqueeze(0).to(torch.float32).to(device)
    inputs = {
        "point_clouds": point_clouds,
        "point_cloud_dims_min": torch.from_numpy(point_cloud.min(axis=0)).unsqueeze(0).to(torch.float32).to(device),
        "point_cloud_dims_max": torch.from_numpy(point_cloud.max(axis=0)).unsqueeze(0).to(torch.float32).to(device),
        "tid": torch.from_numpy(np.array([tid])).to(torch.int32).to(device),
    }
    outputs = model(inputs)
    config_dict = get_ap_config_dict(remove_empty_box=True,
                                     dataset_config=dataset_config,
                                     nms_iou=0.25,
                                     per_class_proposal=False,
                                     use_cls_confidence_only=False)
    m = 1 / float(transformation[0,0])
    centroid = click_point.to_numpy()
    return parse_output(point_clouds, outputs["outputs"], config_dict, centroid, m)  # type: ignore

if __name__ == '__main__':
    from algorithm_assistant import visualizer
    # mesh_file = Path("/media/8TB/dataset/20231214/670384患者姓名石柳/upper_jaw.ply") # 普通数据
    mesh_file = Path("/media/8TB/dataset/20241218/from_yuqi_20241111_extract_crown_finish_s2023-12-18_00002-018/upper_jaw.ply") # 牙冠设计数据
    # mesh_file = Path("/media/8TB/dataset/20230313/13292_649486/lower_jaw.stl") # 原始网格
    mesh = TriangleMesh.from_file(mesh_file)
    visualizer.add_triangle_mesh(triangle_mesh=mesh_file, name="网格")
    click_point = visualizer.get_point(name="F")
    visualizer.add_points([click_point], name="F")
    tooth_detect_results = infer(mesh=mesh, click_point=click_point, tid=13)
    for tooth_detect_result in tooth_detect_results:
        visualizer.add_tooth_detect_result(detect_result=tooth_detect_result, show_keypoints=True, show_axis=True)
    visualizer.show(block= False)


