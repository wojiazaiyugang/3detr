"""
扫描牙齿点击检测
1、检测牙齿bbox
2、检测牙齿轴向、关键点
3、分割
"""
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import DracoPy
import igl
import numpy as np
import requests
from algorithm_assistant import run_func_in_new_process, TriangleMesh, Point3D, logger, ToothDetectResult3D, TOOTH, Vector3D
from xiaoliualgorithmhelper import AlgorithmNode

retain_gpu_model = str(os.environ.get("retain_gpu_model")).upper() == "TRUE"

def run(task_id: str, work_dir: Path, args: Dict[str, Any]) -> Any:
    logger.info(f"接收到任务 {task_id=}")
    try:
        if retain_gpu_model:
            return process_request(work_dir=work_dir, args=args)
        else:
            # noinspection PyArgumentList
            return run_func_in_new_process(process_request, 20, work_dir, args)
    except Exception as err:
        logger.exception(err)
        raise Exception(f"算法处理异常 {err=}")

def process_request(work_dir: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理请求
    :param work_dir:
    :param args:
    :return:
    """
    mesh_data = json.loads(Path(args["mesh_file"]).read_text())
    m = DracoPy.decode(base64.b64decode(mesh_data["mesh_buffer"]))
    mesh = TriangleMesh(vertices=m.points, triangles=m.faces)
    click_point = Point3D(*args["marker"])
    tid = int(args["tid"])
    segmentation = args.get("segmentation", {}).get("enable", False)
    logger.info(f"{len(mesh.vertices)=} {len(mesh.triangles)=} {click_point=} {tid=} {segmentation=}")
    tooth_detect_result, crown = process(mesh=mesh, click_point=click_point, tid=tid, segmentation=segmentation)
    result: Dict[str, Any]
    result = {
        "bbox": [tooth_detect_result.bbox.point1.to_tuple(), tooth_detect_result.bbox.point7.to_tuple()],
        "axis": {key: [value["x"], value["y"], value["z"]] for key, value in tooth_detect_result.tooth_axis.to_dict().items()},
        "keypoints": {key: [value["x"], value["y"], value["z"]] for key, value in tooth_detect_result.tooth_keypoints.to_dict().items()},
    }
    if crown:
        save_file = work_dir.joinpath("crown.drc")
        crown.save(file=save_file)
        result.update({
            "crown": save_file
        })
    return result

def process(mesh: TriangleMesh, click_point: Point3D, tid: int, segmentation: bool) -> Tuple[ToothDetectResult3D, Optional[TriangleMesh]]:
    """
    处理函数
    :param mesh: 15mm剪裁之后的网格
    :param click_point: 点击点
    :param tid: 牙齿tid
    :param segmentation: 是否分割牙冠
    :return:
    """
    from models.inference import infer

    tooth_detect_results = infer(mesh=mesh, click_point=click_point, tid=tid)
    assert len(tooth_detect_results) == 1, f"检测牙齿数量异常 {len(tooth_detect_results)=}"
    tooth_detect_result = tooth_detect_results[0]
    tooth = TOOTH.get_tooth_by_tid(tid=tid)

    # 坐标轴处理成单位向量且正交
    tooth_detect_result.tooth_axis.axisie = tooth_detect_result.tooth_axis.axisie.norm()
    tooth_detect_result.tooth_axis.axismd = tooth_detect_result.tooth_axis.axismd.norm()
    axisfl = np.cross(tooth_detect_result.tooth_axis.axisie.to_numpy(), tooth_detect_result.tooth_axis.axismd.to_numpy())
    tooth_detect_result.tooth_axis.axisfl = Vector3D.from_numpy(axisfl / np.linalg.norm(axisfl))
    axismd = np.cross(tooth_detect_result.tooth_axis.axisfl.to_numpy(), tooth_detect_result.tooth_axis.axisie.to_numpy())
    tooth_detect_result.tooth_axis.axismd = Vector3D.from_numpy(axismd / np.linalg.norm(axismd))
    if tooth.is_area1_tooth or tooth.is_area2_tooth:
        tooth_detect_result.tooth_axis.axisfl = tooth_detect_result.tooth_axis.axisfl.inverse()

    keypoints = tooth_detect_result.tooth_keypoints
    if tooth.is_area1_tooth or tooth.is_area4_tooth:
        # 对调1区和4区的部分关键点
        if hasattr(keypoints, "lmc") and hasattr(keypoints, "ldc"):
            keypoints.lmc, keypoints.ldc = keypoints.ldc, keypoints.lmc
        if hasattr(keypoints, "occm") and hasattr(keypoints, "occd"):
            keypoints.occm, keypoints.occd = keypoints.occd, keypoints.occm
        if hasattr(keypoints, "bmc") and hasattr(keypoints, "bdc"):
            keypoints.bmc, keypoints.bdc = keypoints.bdc, keypoints.bmc
        if hasattr(keypoints, "mrm") and hasattr(keypoints, "mrd"):
            keypoints.mrm, keypoints.mrd = keypoints.mrd, keypoints.mrm
        if hasattr(keypoints, "fcm") and hasattr(keypoints, "fcd"):
            keypoints.fcm, keypoints.fcd = keypoints.fcd, keypoints.fcm
        if hasattr(keypoints, "nfcm") and hasattr(keypoints, "nfcd"):
            keypoints.nfcm, keypoints.nfcd = keypoints.nfcd, keypoints.nfcm

    if segmentation:
        request_data = {
            "scan_mesh": {
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.triangles.tolist()
            },
            "ensemble": True,
            "teeth_bboxes": [{
                "tid": str(tid),
                "bbox": [tooth_detect_result.bbox.point1.to_tuple(), tooth_detect_result.bbox.point7.to_tuple()]
            }]
        }
        scan_tooth_segment_url = os.environ["SCAN_TOOTH_SEGMENT_URL"]
        logger.info(f"请求牙齿分割算法 {scan_tooth_segment_url=}")
        try:
            response = requests.post(scan_tooth_segment_url, json=request_data, timeout=30)
            teeth_crown_meshes = response.json()["teeth_crown_meshes"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求牙齿分割算法失败，可能是没有检测到牙齿 {e=}")
        assert len(teeth_crown_meshes) == 1, f"分割结果异常 {len(teeth_crown_meshes)=}"
        crown_v, crown_f = teeth_crown_meshes[0]["mesh"]["vertices"], teeth_crown_meshes[0]["mesh"]["faces"]
        if len(crown_v) == 0 or len(crown_f) == 0:
            raise RuntimeError(f"牙冠分割结果异常 {len(crown_v)=} {len(crown_f)=}")
        crown = TriangleMesh(vertices=np.array(crown_v, np.float64).reshape(-1, 3), triangles=np.array(crown_f, np.int32).reshape(-1, 3))
        tooth_detect_result.bbox = crown.get_axis_aligned_bounding_box()
    else:
        crown = None
    logger.info(f"关键点投影到牙冠分割结果/原始网格上")
    points = None
    for key in sorted(keypoints.get_all_name()):
        if points is None:
            points = np.array([getattr(keypoints, key).to_numpy()])
        else:
            points = np.vstack((points, np.array([getattr(keypoints, key).to_numpy()])))
    align_mesh = crown if crown else mesh
    _, _, target_points = igl.point_mesh_squared_distance(points,
                                                          np.asarray(align_mesh.vertices),
                                                          np.asarray(align_mesh.triangles))
    for i, key in enumerate(sorted(keypoints.get_all_name())):
        setattr(keypoints, key, Point3D.from_numpy(target_points[i]))
    tooth_detect_result.tooth_keypoints = keypoints
    return tooth_detect_result, crown


if __name__ == '__main__':
    # os.environ["SCAN_TOOTH_SEGMENT_URL"] = "http://127.0.0.1:5000/scan_tooth_segment"

    node = AlgorithmNode(name="scan_tooth_marker_detection",
                         user_name="algorithm_node",
                         password="algorithm_node",
                         # host_name="yujiannan's PC",
                         # version="20250520",
                         kwargs={},
                         func=run)
    node.run()

    # from algorithm_assistant import visualizer, COLOR
    #
    #
    # mesh_file = Path("/media/8TB/dataset/20231214/670384患者姓名石柳/upper_jaw.ply") # 牙冠设计数据
    # mesh = TriangleMesh.from_file(mesh_file)
    # visualizer.add_triangle_mesh(triangle_mesh=mesh, name="网格", color=COLOR.body_fat)
    # try:
    #     click_point = visualizer.get_point(name="F")
    # except KeyError:
    #     click_point = Point3D(0, 0, 0)
    # visualizer.add_points([click_point], name="F")
    # tooth_detect_result, crown = process(mesh=mesh, click_point=click_point, tid=14, segmentation=True)
    # visualizer.add_tooth_detect_result(detect_result=tooth_detect_result, show_keypoints=True, show_axis=True)
    # if crown:
    #     visualizer.add_triangle_mesh(triangle_mesh=crown, name="分割", color=COLOR.tissue)
    # visualizer.show(block= False)