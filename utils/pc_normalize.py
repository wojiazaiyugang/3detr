import numpy as np

from algorithm_assistant import TriangleMesh, Point3D, Transformation


def get_normalize_transformation(mesh: TriangleMesh, click_point: Point3D) -> Transformation:
    """
    网格归一化
    :param mesh:
    :param click_point: 点击点
    :return:
    """
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = -1 * click_point.to_numpy()

    scale = 1 / np.max(np.abs(mesh.vertices - click_point.to_numpy()))
    scale_matrix = np.identity(4)
    scale_matrix[0,0], scale_matrix[1,1], scale_matrix[2,2] = scale, scale, scale

    return scale_matrix @ translation_matrix


