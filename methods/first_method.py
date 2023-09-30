import cv2
import numpy as np

image = cv2.imread("../test_pictures/lssd9.jpg", cv2.IMREAD_COLOR)
image_shape = image.shape[:2]
cv2.imshow("Original", image)

blue,green,red = cv2.split(image)

# converts the image to lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# splits the channels of the image
light_level, a_level, b_level = cv2.split(lab_image)

# calculates the mean and standard deviation of each channel in the LAB colour space
light_level_mean, light_level_stddev = cv2.meanStdDev(light_level)
a_level_mean, a_level_stddev = cv2.meanStdDev(a_level)
b_level_mean, b_level_stddev = cv2.meanStdDev(b_level)

# creates a blank mask with the shape of the image
shadow_mask = np.zeros(image_shape, np.uint8)

if a_level_mean + b_level_mean <= 256:
    shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3)] = 255
else:
    shadow_mask[(light_level <= light_level_mean - light_level_stddev/3) & (b_level <= b_level_mean - b_level_stddev/3) & (b_level >= -1*b_level_mean + b_level_stddev/3)] = 255

# using morphological opening to erase smaller non-shadow pixels
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

dst = cv2.dilate(shadow_mask, struct)
dst2 = cv2.erode(dst, struct, iterations=2)
mask = cv2.dilate(dst2, struct)

# using median blur to smoothen rough edges
mask = cv2.medianBlur(mask, 3)

# finding contours with mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# filtering out the smaller non-shadow contours
corrected_contours = []
for contour in contours:
    if cv2.arcLength(contour, True) > 60:
        corrected_contours.append(contour)

# TODO maybe there is a simple way to separate regions
# based on the original mask, creating a better one to fill in smaller gaps using the found contours
contour_mask = np.zeros(image_shape, np.uint8)

individual_shadow_masks = []
individiual_shadow_region_mean = []

individual_shadow_contour_mask = []
individual_shadow_contour_mask_mean = []

for contour in corrected_contours:

    temp_mask = np.zeros(image_shape, np.uint8)

    # calculating the outside mean values of the contour
    cv2.drawContours(temp_mask, [contour], 0, 255, 1)
    individual_shadow_contour_mask_mean = cv2.mean(image,temp_mask)
    individual_shadow_contour_mask.append(temp_mask)

    # remaking the shadow mask
    cv2.drawContours(temp_mask,[contour], 0, 255, -1)
    individual_shadow_masks.append(temp_mask)

    # calculating the color mean inside each shadow region
    temp_mean = cv2.mean(image,temp_mask)
    individiual_shadow_region_mean.append(temp_mean)

    cv2.drawContours(contour_mask, [contour], 0, 255, -1)

    # calculating the color ratio inside and outside of shadow region
    blue_ratio = round(individual_shadow_contour_mask_mean[0] / temp_mean[0],4)
    green_ratio = round(individual_shadow_contour_mask_mean[1] / temp_mean[1],4)
    red_ratio = round(individual_shadow_contour_mask_mean[2] / temp_mean[2],4)

    # getting the shadow part of the image
    temp_image = image.copy()
    temp_image[temp_mask != 255] = 0

    temp_blue,temp_green,temp_red = cv2.split(temp_image)

    # TODO solve how to multiply each channel

    temp_blue = cv2.multiply(temp_blue,blue_ratio)
    temp_green = cv2.multiply(temp_green,green_ratio)
    temp_red = cv2.multiply(temp_red,red_ratio)

    temp_image = cv2.merge((temp_blue,temp_green,temp_red))

    result = cv2.add(image,temp_image)



    #TODO
    # calculate the mean of the outline contour with the given matrix calculation stuff
    # get the outside/inside region ratio, put it in the array
    # remove the current shadow on the final image
    
    # maybe one array needed of of contour outlines
    #TODO add to array and calculate inside of region


cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
