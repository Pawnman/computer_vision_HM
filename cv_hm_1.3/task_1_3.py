import cv2
import numpy as np

def new_borders(M, h: int, w: int):
    """
    Вычисление новых границ окна изображения

    Args:
        M (np.ndarray): Матрица аффинного преобразования (2x3).
        h (int): Высота исходного изображения.
        w (int): Ширина исходного изображения.
    """
    # Крайние точки исходного изображения
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
    transformed_corners = M @ corners.T

    # Получаем новые границы
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])

    new_w = int(np.round(max_x - min_x))
    new_h = int(np.round(max_y - min_y))

    return (min_x, min_y), (new_w, new_h)

def rotate(image: np.ndarray, angle: float, point: tuple = None) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """

    h, w = image.shape[:2]
    
    # Создание матрицы поворота вокруг point
    rotation_matrix = cv2.getRotationMatrix2D(point, angle, 1.0)

    low, new_shp = new_borders(rotation_matrix, h, w)
    rotation_matrix[0][-1] -= low[0]
    rotation_matrix[1][-1] -= low[1]

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_shp[0], new_shp[1]))

    return rotated_image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    # Получаем матрицу перспективного преобразования
    matrix_convert = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))

    # Определяем размер нового изображения
    width = int(max(points2[0][0], points2[1][0], points2[2][0], points2[3][0]))
    height = int(max(points2[0][1], points2[1][1], points2[2][1], points2[3][1]))

    # Применяем перспективное преобразование
    transformed_image = cv2.warpPerspective(image, matrix_convert, (width, height))

    return transformed_image