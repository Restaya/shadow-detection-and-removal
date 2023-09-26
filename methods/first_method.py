import cv2
import numpy as np

image = cv2.imread("../test_pictures/lssd803.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Image", image)

# converts the image to lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# splits the channels of the image
light_level, a_level, b_level = cv2.split(lab_image)

# calculates the mean and standard deviation of each channel in the LAB colour space
light_level_mean, light_level_stddev = cv2.meanStdDev(light_level)
a_level_mean, a_level_stddev = cv2.meanStdDev(a_level)
b_level_mean, b_level_stddev = cv2.meanStdDev(b_level)

# combining the Light and B channels
merged_l_b = cv2.add(light_level,b_level)
merged_channels_mean, merged_channels_stddev = cv2.meanStdDev(merged_l_b)

# creates a blank mask with the shape of the image
shadow_mask = np.zeros(lab_image.shape[:2], np.uint8)

if a_level_mean + b_level_mean <= 256:
    shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3)] = 255
else:
    #TODO this threshhold is temporary, not sure which is the optimal
    #shadow_mask[merged_l_b <= merged_channels_mean - merged_channels_stddev/3] = 255
    #shadow_mask[(light_level <= light_level_mean - light_level_stddev/3) & (b_level <= b_level_mean - b_level_stddev/3)] = 255
    shadow_mask[(light_level <= light_level_mean - light_level_stddev/3) & (b_level <= b_level_mean - b_level_stddev/3) & (b_level >= -1*b_level_mean + b_level_stddev/3)] = 255

# using morphological opening to erase smaller non-shadow pixels
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

#TODO not sure which morphological operation is the optimal

dst = cv2.dilate(shadow_mask, struct)
dst2 = cv2.erode(dst, struct, iterations=2)
mask = cv2.dilate(dst2, struct)

#mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, struct)

mask = cv2.medianBlur(mask,3)

# TODO calculate for each shadow region separately
# calculating the average values within the detected shadow regions
image_mean = cv2.mean(image, mask)

blue_mean = image_mean[0]
green_mean = image_mean[1]
red_mean = image_mean[2]

lab_image = cv2.cvtColor(lab_image,cv2.COLOR_LAB2BGR)
lab_image[mask != 0] = 255

cv2.imshow("Result", lab_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
