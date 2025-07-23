"""
牙齿检测
"""
import time
import os
from typing import Dict, Any, List, Optional

import numpy as np
from flask import request, Flask
from algorithm_assistant import logger, run_func_in_new_process, TOOTH, TriangleMesh, Vector3D

app = Flask(__name__)

@app.route("/scan_tooth_detect", methods=["POST"])
def process_view() -> Dict[str, Any]:
    retain_gpu_model = str(os.environ.get("retain_gpu_model")).upper() == "TRUE"  # 是否在GPU上保留模型
    logger.info(f"接收到请求, {retain_gpu_model=}")
    start_time = time.time()
    if not retain_gpu_model:
        data = run_func_in_new_process(process, 20, request.json)
        logger.info(f"""检测进程执行{"成功" if data is not None else "失败"}，耗时{time.time() - start_time}秒""")
        return {"teeth_bboxes": data}
    else:
        result = {"teeth_bboxes": process(request.json)}  # type: ignore
        logger.info(f"检测执行完毕，耗时{time.time() - start_time}秒")
        return result


def process(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    处理检测任务
    :param data:
    :return:
    """
    from models.inference import infer

    vertices, faces = np.array(data["scan_mesh"]["vertices"]), np.array(data["scan_mesh"]["faces"])
    logger.info(f"开始处理检测任务，{vertices.shape=}, {faces.shape=}")
    mesh = TriangleMesh(vertices=vertices, triangles=faces)
    detect_results = infer(mesh=mesh)
    for tooth_detect_result in detect_results:
        tooth = TOOTH.get_tooth_by_category(category=tooth_detect_result.category)
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

    # 只返回需要的字段
    response_data = []
    for detect_result in detect_results:
        bbox = detect_result.bbox
        assert detect_result.category and detect_result.tooth_keypoints and detect_result.tooth_axis
        tooth = TOOTH.get_tooth_by_category(detect_result.category)
        response_data.append({
            "tid": str(tooth.tid),
            "bbox": [list(bbox.point1.to_tuple()), list(bbox.point7.to_tuple())],
            "tooth_keypoints": detect_result.tooth_keypoints.to_dict(),
            "tooth_axis": detect_result.tooth_axis.to_dict(),
            "score": detect_result.score
        })
    return response_data


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)
