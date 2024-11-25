import cv2

import numpy as np

def area_recognition(image_path):
    """

    :param image_path: путь к изображению
    :return: строка 'Forest' или 'Desert'
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Диапазоны для зелёного (лес)
    green_low = (35, 40, 40)  # Оттенки зелёного
    green_high = (85, 255, 255)

    # Диапазоны для жёлтого/коричневого (пустыня)
    yellow_low = (15, 40, 40)  # Оттенки жёлтого/коричневого
    yellow_high = (35, 255, 255)

    # Создание масок
    green_mask = cv2.inRange(image_hsv, green_low, green_high)
    yellow_mask = cv2.inRange(image_hsv, yellow_low, yellow_high)

    # Подсчёт пикселей, попавших в диапазоны оттенков зеленого и желтого
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Классификация изображений
    if green_pixels > yellow_pixels:
        return "Forest"
    else:
        return "Desert"
