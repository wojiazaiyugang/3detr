import base64
import json
from pathlib import Path

import DracoPy
import requests
from algorithm_assistant import visualizer, Point3D, Sphere, TriangleMesh, BBox3D, ToothAxis, TOOTH
from xiaoliualgorithmhelper import AlgorithmClient


def mesh_to_b64(mesh: TriangleMesh) -> str:
    """
    将mesh转换为b64
    :param mesh:
    :return:
    """
    buffer = DracoPy.encode(
        mesh.vertices, faces=mesh.triangles,
        quantization_bits=14, compression_level=1,
        quantization_range=-1, quantization_origin=None,
        create_metadata=False, preserve_order=False,
    )
    data = base64.b64encode(buffer).decode("utf-8")
    return data

if __name__ == '__main__':
    mesh_file = Path("/media/8TB/dataset/20231214/670384患者姓名石柳/upper_jaw.ply") # 普通数据
    mesh = TriangleMesh.from_file(mesh_file)
    visualizer.add_triangle_mesh(triangle_mesh=mesh_file, name="网格")
    try:
        click_point = visualizer.get_point(name="F")
    except KeyError:
        click_point = Point3D(-6.057, 0.332, 28.141)
    visualizer.add_points([click_point], name="F")
    clip_mesh = mesh.crop_by_sphere(sphere=Sphere(center=click_point, radius=15))
    assert clip_mesh is not None
    mesh_file = Path(__file__).parent.joinpath("temp.drc")
    clip_mesh.save(mesh_file)
    client = AlgorithmClient(user_name="test_user", password="test_password")
    tooth = TOOTH.tooth_11
    save_mesh_suffix = ".ply"
    result = client.submit_task(name="scan_tooth_marker_detection",
                       args={
                           "marker": click_point.to_tuple(),
                           "tid": tooth.tid,
                           "mesh_file": mesh_file,
                           "segmentation": {
                               "enable": True
                           },
                           "save_mesh_suffix": save_mesh_suffix
                       })
    mesh_file.unlink()
    task_result = client.get_task_result(task_id=result["task_id"], timeout=-1)["result"]
    bbox = BBox3D(point1=Point3D(*task_result["bbox"][0]), point7=Point3D(*task_result["bbox"][1]))
    tooth_axis = ToothAxis.from_dict({key: {"x": value[0], "y": value[1], "z": value[2]} for key, value in task_result["axis"].items()})
    tooth_keypoints = tooth.keypoints_type.from_dict({key: {"x": value[0], "y": value[1], "z": value[2]} for key, value in task_result["keypoints"].items()})
    visualizer.add_bbox(bbox=bbox)
    visualizer.add_tooth_axis(axis=tooth_axis, point=tooth_keypoints.occc)
    visualizer.add_tooth_keypoints(tooth_keypoints=tooth_keypoints, tooth=tooth)
    if "crown" in task_result:
        crown_file = Path(__file__).parent.joinpath(f"crown{save_mesh_suffix}")
        crown_file.write_bytes(requests.get(task_result["crown"]).content)
        visualizer.add_triangle_mesh(triangle_mesh=crown_file, name="分割结果")
        crown_file.unlink()
    visualizer.show(block=False)