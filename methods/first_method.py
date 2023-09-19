import cv2

image = cv2.imread("../test_pictures/first_method_optimal.jpg",cv2.IMREAD_COLOR)

# converts the image to lab color space
lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)

# splits the channels of the image
light_level, a_level,b_level = cv2.split(lab_image)

# calculates the mean of each channel in the LAB colour space
light_level_mean = cv2.mean(lab_image)[0]
a_level_mean = cv2.mean(lab_image)[1]
b_level_mean = cv2.mean(lab_image)[2]

# calculates the standard deviation for later usage
light_level_stddev = cv2.meanStdDev(light_level_mean)[0]

shadow_mask = lab_image.copy()
shadow_mask.fill(0)

# TODO if a_level_mean + b_level_mean <= 256:
shadow_mask[light_level <= (light_level_mean - light_level_stddev/3)] = 255
#TODO else:


# using morphological opening to erase smaller non-shadow pixels
struct = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

mask_open = cv2.morphologyEx(shadow_mask,cv2.MORPH_OPEN,struct)

# try gaussian blur then mask it

cv2.imshow("Result",mask_open)
cv2.imshow("Image",image)

cv2.waitKey(0)
cv2.destroyAllWindows()