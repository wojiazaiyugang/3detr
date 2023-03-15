import numpy as np


def angle_between(v1, v2, angle: bool = False) -> float:
    """
    计算两个向量之间的夹角
    :param v1:
    :param v2:
    :param angle: 是否返回角度，否则返回弧度
    :return:
    """
    unit_vector = lambda v: v / np.linalg.norm(v)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    scale = 180 / np.pi if angle else 1
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * scale
