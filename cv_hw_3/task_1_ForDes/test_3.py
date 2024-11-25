import cv2

from recognition import area_recognition


for i in range(17):
    image = cv2.imread(f"cv_hw_3/task_1_ForDes/img/test_image_{i}.jpg")
    result = area_recognition(f"cv_hw_3/task_1_ForDes/img/test_image_{i}.jpg")
    cv2.imshow(f"{result}", image)
    cv2.waitKey(0)
