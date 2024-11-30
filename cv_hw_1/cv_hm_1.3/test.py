import cv2
import numpy as np

def apply_affine_transform_with_resize(image, points1, points2):
    """
    Применяет аффинное преобразование к изображению с учётом изменения размера, чтобы изображение не обрезалось.
    
    Args:
        image (np.ndarray): Исходное изображение.
        points1 (np.ndarray): Исходные точки (3 точки).
        points2 (np.ndarray): Точки назначения (3 точки).
    
    Returns:
        np.ndarray: Преобразованное изображение.
    """
    # Получаем матрицу аффинного преобразования
    M = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
    
    # Определяем размеры исходного изображения
    h, w = image.shape[:2]
    
    # Вычисляем новые координаты углов изображения
    corners = np.array([
        [0, 0],         # Верхний левый угол
        [w, 0],         # Верхний правый угол
        [0, h],         # Нижний левый угол
        [w, h]          # Нижний правый угол
    ])
    
    # Преобразуем углы с использованием матрицы M
    transformed_corners = cv2.transform(np.array([corners], dtype=np.float32), M)[0]
    
    # Находим минимальные и максимальные координаты
    min_x = int(np.min(transformed_corners[:, 0]))
    min_y = int(np.min(transformed_corners[:, 1]))
    max_x = int(np.max(transformed_corners[:, 0]))
    max_y = int(np.max(transformed_corners[:, 1]))
    
    # Вычисляем новые размеры изображения
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    # Создаём смещение, чтобы изображение не выходило за границы
    offset = np.array([
        [-min_x, -min_y],  # Смещение для углов
        [-min_x, -min_y],
        [-min_x, -min_y]
    ])
    
    # Корректируем матрицу аффинного преобразования с учётом смещения
    M_with_offset = cv2.getAffineTransform(
        np.float32(points1),
        np.float32(points2) + offset
    )
    
    # Применяем аффинное преобразование
    transformed_image = cv2.warpAffine(image, M_with_offset, (new_width, new_height))
    
    return transformed_image

# Пример использования
if __name__ == "__main__":
    # Загрузка изображения
    test_image = cv2.imread("cv_hw_1\cv_hm_1.3\img\lk.jpg")

    # Точки для преобразования
    points1 = np.array([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
    points2 = np.array([[70, 70], [220, 30], [100, 250]], dtype=np.float32)  # Точки назначения
    

    # Преобразование
    result = apply_affine_transform_with_resize(test_image, points1, points2)
    
    cv2.imshow("Original", result)
    cv2.waitKey(0)
    # Отображение результата
    cv2.imshow("Original", test_image)
    cv2.imshow("Transformed", result)
    cv2.waitKey(0)
   