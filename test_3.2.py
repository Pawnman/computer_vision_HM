import cv2
import matplotlib.pyplot as plt
import numpy as np

from task_3 import apply_warpAffine

image = cv2.imread('img/notebook.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

hsv_low = (0.0 * 360, 60, 60)
hsv_high = (0.02 * 360, 250, 255)

area = cv2.inRange(image_hsv, hsv_low, hsv_high)

contours, _ = cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (255, 0, 0), 2)

contoured_image_only = np.zeros_like(image)
cv2.drawContours(contoured_image_only, contours, -1, (255, 0, 0), 2)



if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    for point in approx:
        cv2.circle(contoured_image, tuple(point[0]), 10, (0, 255, 0), -1)

    if len(approx) == 4:
        points = approx.reshape(4, 2)

        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

        destination_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                                      dtype='float32')

        cropped_image = apply_warpAffine(image, points, destination_points)

        fig, m_axs = plt.subplots(1, 4, figsize=(12, 12))
        ax1, ax2, ax3, ax4 = m_axs

        ax1.set_title('Original Image', fontsize=15)
        ax1.imshow(image)
        ax2.set_title('Image mask', fontsize=15)
        ax2.imshow(area, cmap='gray')
        ax3.set_title('Image with points', fontsize=15)
        ax3.imshow(contoured_image)
        ax4.set_title('Result image', fontsize=15)
        ax4.imshow(cropped_image)

        plt.show()
    else:
        print("Не удалось найти 4 угла документа.")
else:
    print("Контуры не найдены.")