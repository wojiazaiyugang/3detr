"""
扫描牙齿点击检测
1、检测牙齿bbox
2、检测牙齿轴向、关键点
3、分割
"""
import os
from pathlib import Path
from typing import Dict, Any

import requests

from xiaoliualgorithmhelper import AlgorithmNode
from algorithm_assistant import run_func_in_new_process, TriangleMesh, Point3D, logger
from models.inference import infer

retain_gpu_model = str(os.environ.get("retain_gpu_model")).upper() == "TRUE"

def process(mesh: TriangleMesh, click_point: Point3D, segmentation: bool):
    """
    :param mesh: 扫描牙齿网格
    :param click_point: 点击点
    :param segmentation: 是否分割
    :return:
    """
    tooth_detect_results = infer(mesh=mesh, click_point=click_point)
    assert len(tooth_detect_results) == 1
    tooth_detect_result = tooth_detect_results[0]
    if segmentation:
        scan_tooth_marker_segment_url = os.environ["scan_tooth_marker_segment_url"]
        logger.info(f"请求单牙分割算法 {scan_tooth_marker_segment_url=}")

    return tooth_detect_result

def run(task_id: str, work_dir: Path, args: Dict[str, Any]) -> Any:
    try:



        mesh = TriangleMesh()
        click_point = Point3D()
        if retain_gpu_model:
            return process(mesh=mesh, click_point=click_point)
        else:
            return run_func_in_new_process(func=process, timeout=20, args=(mesh, click_point))
    except Exception as err:
        logger.exception(err)
        raise Exception(f"算法处理异常 {err=}")



if __name__ == '__main__':
    node = AlgorithmNode(name="scan_tooth_marker_detection",
                         user_name="algorithm_node",
                         password="algorithm_node",
                         host_name="yujiannan's PC",
                         version="",
                         kwargs={},
                         func=run)
    node.run()
