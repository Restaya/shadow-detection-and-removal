import cv2

image = cv2.imread("../test_pictures/first_method_optimal.jpg")

lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)

light_level, a_level,b_level = cv2.split(lab_image)

cv2.imshow("Image",lab_image)


cv2.waitKey(0)
cv2.destroyAllWindows()