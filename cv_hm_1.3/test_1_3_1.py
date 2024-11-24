import cv2

from task_1_3 import rotate

test_image = cv2.imread('cv_hm_1.3/img/testrose.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
cv2.imshow("Original image", test_image)

cv2.waitKey(0)

test_point = (100, 200)
test_angle = 30

transformed_image = rotate(test_image, test_angle, test_point)
cv2.imshow("Rotated image", transformed_image)

cv2.waitKey(0)
