import cv2

from task_2 import find_road_number

test_image = cv2.imread('cv_hm_1.2\img\image_02.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

cv2.imshow("Picture", test_image)
cv2.waitKey(0)

road_number = find_road_number(test_image)

print(f'Нужно перестроиться на дорогу номер {road_number}')
