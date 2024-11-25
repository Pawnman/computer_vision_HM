import cv2
import numpy as np

# Загрузка изображений
source_image = cv2.imread("cv_hw_3/img_faces/face-1.jpg")  # Лицо, откуда берем части
target_image = cv2.imread("cv_hw_3/img_faces/face_2.jpg")  # Лицо, куда вставляем части

# Определение области для глаз, носа и рта на исходном изображении
# Здесь область задается прямоугольниками (x, y, w, h)
# Их можно определить вручную с помощью любого инструмента или встроенных методов OpenCV
eye_region = (50, 60, 100, 50)  # Пример: x, y, width, height
nose_region = (80, 120, 60, 60)
mouth_region = (70, 200, 120, 40)

# Создание маски для каждой области
def create_mask(image, region):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x, y, w, h = region
    mask[y:y+h, x:x+w] = 255
    return mask

eye_mask = create_mask(source_image, eye_region)
nose_mask = create_mask(source_image, nose_region)
mouth_mask = create_mask(source_image, mouth_region)

# Объединение масок
combined_mask = cv2.add(eye_mask, cv2.add(nose_mask, mouth_mask))

# Центр вставки (пример: центр изображения)
center = (target_image.shape[1] // 2, target_image.shape[0] // 2)

# Применение Poisson Image Editing
output = cv2.seamlessClone(
    source_image,  # Исходное изображение
    target_image,  # Целевое изображение
    combined_mask,  # Маска
    center,         # Центр вставки
    cv2.NORMAL_CLONE  # Тип клонирования: нормальное смешивание
)

# Сохранение и отображение результата
cv2.imshow("Blended Face", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("blended_face.jpg", output)
