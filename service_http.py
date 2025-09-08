"""
牙齿检测
"""
import os
import time
from typing import Dict, Any, List, Optional

import numpy as np
from algorithm_assistant import logger, run_func_in_new_process, TOOTH, TriangleMesh, Vector3D
from flask import request, Flask

from utils.data_class import ToothDetect

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

def post_process_tooth_detect_results(detect_results: List[ToothDetect], return_meta: bool) -> List[ToothDetect]:
    """
    对牙齿检测结果进行后处理
    1、过滤低置信度的检测结果
    2、处理牙号冲突
    :param detect_results:
    :param return_meta:
    :return:
    """
    # 按照置信度排序
    detect_results = list(sorted(detect_results, key=lambda x: x.score, reverse=True))
    # 过滤阈值
    obj_score = float(os.environ.get("obj_score", 0.81))
    cls_score = float(os.environ.get("cls_score", 0.81))
    scores = [(d.label ,round(d.data["obj_score"], 4), round(d.data["cls_score"], 4)) for d in detect_results]
    logger.info(f"检测到{len(detect_results)}颗牙齿, 过滤前置信度{scores}")
    # 这里应该是and 但是没有看非常多的数据 怕太严格了，先用or
    detect_results = [d for d in detect_results if d.data["obj_score"] > obj_score or d.data["cls_score"] > cls_score]
    if len(detect_results) != len(scores):
        logger.warning(f"过滤后剩余{len(detect_results)}颗牙齿")
    else:
        logger.info(f"过滤后剩余{len(detect_results)}颗牙齿")
    if return_meta:
        # return_meta的时候不进行牙号冲突处理，全量返回
        return detect_results
    # 如果牙号冲突，这里尝试解决，如果第一置信度的牙号已经被占了，就看能不能用第二置信度的牙号
    results = []
    for detect_result in detect_results:
        if detect_result.category not in [result.category for result in results]:
            results.append(detect_result)
        else:
            category_score = detect_result.data["category_score"]
            category_score_list = sorted(category_score.items(), key=lambda x: x[1], reverse=True)
            if category_score_list[1][1] < 0.001:
                continue
            current_tooth = TOOTH.get_tooth_by_category(detect_result.category)
            target_tooth = TOOTH.get_tooth_by_category(category_score_list[1][0])
            if target_tooth.category in [result.category for result in results]:
                logger.warning(f"第一牙号冲突且第二置信度牙号冲突，尝试解决失败，第一牙号{current_tooth.tid}，第二牙号{target_tooth.tid}")
                continue
            if (current_tooth.is_lower_tooth and not target_tooth.is_lower_tooth) or (current_tooth.is_upper_tooth and not target_tooth.is_upper_tooth):
                logger.warning(f"第一牙号冲突，且第二牙号不是同一颌的牙号，尝试解决失败，第一牙号{current_tooth.tid}，第二牙号{target_tooth.tid}")
                continue
            logger.warning(f"第一牙号冲突，使用第二置信度的牙号，第一牙号{TOOTH.get_tooth_by_category(detect_result.category).tid}，第二牙号{target_tooth.tid}")
            detect_result.category = target_tooth.category
            detect_result.label = target_tooth.name
            detect_result.score = detect_result.data["obj_score"] * category_score_list[1][1]
            detect_result.tooth_keypoints = target_tooth.keypoints_type.from_dict(detect_result.data["keypoints"])
            results.append(detect_result)
    return results


def process(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    处理检测任务
    :param data:
    :return:
    """
    from models.inference import infer

    vertices, faces = np.array(data["scan_mesh"]["vertices"]), np.array(data["scan_mesh"]["faces"])
    # 是否返回meta信息，默认为False，如果为True，检测到的box将不会进行去重，同时额外返回全量的关键点数据
    return_meta: bool = data.get("return_meta", False)

    logger.info(f"开始处理检测任务，{return_meta=} {vertices.shape=}, {faces.shape=}")
    mesh = TriangleMesh(vertices=vertices, triangles=faces)
    detect_results = infer(mesh=mesh)
    detect_results = post_process_tooth_detect_results(detect_results=detect_results, return_meta=return_meta)
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
        keypoints, data = tooth_detect_result.tooth_keypoints, tooth_detect_result.data
        if tooth.is_area1_tooth or tooth.is_area4_tooth:
            # 对调1区和4区的部分关键点
            if hasattr(keypoints, "lmc") and hasattr(keypoints, "ldc"):
                keypoints.lmc, keypoints.ldc = keypoints.ldc, keypoints.lmc
                data["keypoints"]["lmc"], data["keypoints"]["ldc"] = data["keypoints"]["ldc"], data["keypoints"]["lmc"]
            if hasattr(keypoints, "occm") and hasattr(keypoints, "occd"):
                keypoints.occm, keypoints.occd = keypoints.occd, keypoints.occm
                data["keypoints"]["occm"], data["keypoints"]["occd"] = data["keypoints"]["occd"], data["keypoints"]["occm"]
            if hasattr(keypoints, "bmc") and hasattr(keypoints, "bdc"):
                keypoints.bmc, keypoints.bdc = keypoints.bdc, keypoints.bmc
                data["keypoints"]["bmc"], data["keypoints"]["bdc"] = data["keypoints"]["bdc"], data["keypoints"]["bmc"]
            if hasattr(keypoints, "mrm") and hasattr(keypoints, "mrd"):
                keypoints.mrm, keypoints.mrd = keypoints.mrd, keypoints.mrm
                data["keypoints"]["mrm"], data["keypoints"]["mrd"] = data["keypoints"]["mrd"], data["keypoints"]["mrm"]
            if hasattr(keypoints, "fcm") and hasattr(keypoints, "fcd"):
                keypoints.fcm, keypoints.fcd = keypoints.fcd, keypoints.fcm
                data["keypoints"]["fcm"], data["keypoints"]["fcd"] = data["keypoints"]["fcd"], data["keypoints"]["fcm"]
            if hasattr(keypoints, "nfcm") and hasattr(keypoints, "nfcd"):
                keypoints.nfcm, keypoints.nfcd = keypoints.nfcd, keypoints.nfcm
                data["keypoints"]["nfcm"], data["keypoints"]["nfcd"] = data["keypoints"]["nfcd"], data["keypoints"]["nfcm"]

    # 只返回需要的字段
    result = []
    for detect_result in detect_results:
        bbox, data = detect_result.bbox, detect_result.data
        assert detect_result.category and detect_result.tooth_keypoints and detect_result.tooth_axis
        tooth = TOOTH.get_tooth_by_category(detect_result.category)
        response_data = {
            "tid": str(tooth.tid),
            "bbox": [list(bbox.point1.to_tuple()), list(bbox.point7.to_tuple())],
            "tooth_keypoints": detect_result.tooth_keypoints.to_dict(),
            "tooth_axis": detect_result.tooth_axis.to_dict(),
            "score": detect_result.score
        }
        if return_meta:
            response_data["keypoints"] = data["keypoints"]
            # noinspection PyTypeChecker
            response_data["category_info"] = list(sorted([
                {
                    "tid": TOOTH.get_tooth_by_category(category).tid,
                    "score": score
                } for category, score in data["category_score"].items() if category != 0
            ], key=lambda x: x["score"], reverse=True))[:5]
        result.append(response_data)
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
