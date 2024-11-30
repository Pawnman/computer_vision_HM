import cv2

import numpy as np

def apply_warpAffine(image, points1, points2):
    """
    Применяет перспективное преобразование к изображению.

    Args:
        image (np.ndarray): Исходное изображение, к которому будет применяться преобразование.
        points1 (list of tuple): Список координат точек в исходном изображении (4 точки).
        points2 (list of tuple): Список координат точек в целевом изображении (4 точки).

    Returns:
        np.ndarray: Преобразованное изображение.
    """
    # Получаем матрицу перспективного преобразования
    M = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))

    # Определяем размер нового изображения
    width = int(max(points2[0][0], points2[1][0], points2[2][0], points2[3][0]))
    height = int(max(points2[0][1], points2[1][1], points2[2][1], points2[3][1]))

    # Применяем перспективное преобразование
    transformed_image = cv2.warpPerspective(image, M, (width, height))

    return transformed_image

test_image = cv2.imread('cv_hw_1\cv_hm_1.3\img\lk.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
cv2.imshow("Original", test_image)
cv2.waitKey(0)

test_point_1 = np.float32([[50, 50], [400, 50], [400, 200], [50, 200]])
test_point_2 = np.float32([[100, 100], [300, 100], [300, 250], [100, 250]])

transformed_image = apply_warpAffine(test_image, test_point_1, test_point_2)
cv2.imshow("Transformed", transformed_image)
cv2.waitKey(0)
