import cv2
import numpy as np

import matplotlib.pyplot as plt

from task_1 import find_way_from_maze

def plot_maze_path(image: np.ndarray, coords: tuple) -> np.ndarray:
    """
    Нарисовать путь через лабиринт на изображении. 
    Вспомогательная функция.
     
    :param image: изображение лабиринта
    :param coords: координаты пути через лабиринт типа (x, y) где x и y - массивы координат точек
    :return img_wpath: исходное изображение с отрисованными координатами 
    """
    if image.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    img_wpath = image.copy()
    if coords:
        x, y = coords
        img_wpath[x, y, :] = [0, 0, 255]

    return img_wpath

test_image = cv2.imread('cv_hm_1.1/img/30 by 30 orthogonal maze.png')  # загрузить тестовую картинку

cv2.imshow("Lab",test_image)
cv2.waitKey(0)

way_coords = find_way_from_maze(test_image)  # вычислить координаты пути через лабиринт

image_with_way = plot_maze_path(test_image, way_coords)

cv2.imshow("Result", image_with_way)
cv2.waitKey(0)