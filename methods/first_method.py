import cv2
import numpy as np

image = cv2.imread("../test_pictures/lssd9.jpg", cv2.IMREAD_COLOR)
image_shape = image.shape[:2]
cv2.imshow("Original", image)

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
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# creating external contours
contours_external, hierarchy_external = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# filtering out the smaller non-shadow contours
corrected_contours = []
for contour in contours:
    if cv2.arcLength(contour, True) > 60:
        corrected_contours.append(contour)

# same for external contours
corrected_external_contours = []
for contour in contours_external:
    if cv2.arcLength(contour, True) > 60:
        corrected_external_contours.append(contour)

#TODO delete when done
print(len(corrected_contours) == len(corrected_external_contours))

# based on the original mask, creating a better one to fill in smaller gaps using the found contours
contour_mask = np.zeros(image_shape, np.uint8)

# inside shadow region masks
inside_shadow_masks = []

# external contours masks
external_contour_masks = []

# creating a better mask with contours
sc_mask = np.zeros(image_shape, np.uint8)

for i in range(len(corrected_contours)):

    temp_shadow_region_mask = np.zeros(image_shape, np.uint8)
    temp_external_contour_mask = np.zeros(image_shape, np.uint8)

    cv2.drawContours(sc_mask,corrected_contours,i,255,-1)

    # calculating the mean color values inside the shadow region
    cv2.drawContours(temp_shadow_region_mask,corrected_contours,i,255,-1)
    temp_shadow_mean = cv2.mean(image,temp_shadow_region_mask)
    inside_shadow_masks.append(temp_shadow_region_mask)

    # maybe do this -> round(temp_shadow_mean[0],0)

    # calculating the outside mean color values
    cv2.drawContours(temp_external_contour_mask,corrected_external_contours,i,255,3)
    temp_ec_mask_mean = cv2.mean(image,temp_external_contour_mask)
    external_contour_masks.append(temp_external_contour_mask)

    # calculating the ratios between outside/inside
    blue_ratio = temp_ec_mask_mean[0] / temp_shadow_mean[0]
    green_ratio = temp_ec_mask_mean[1] / temp_shadow_mean[1]
    red_ratio = temp_ec_mask_mean[2] / temp_shadow_mean[2]

    temp_image = image.copy()
    temp_image[temp_shadow_region_mask != 255] = 0
    #image[temp_shadow_region_mask == 255] = 0

    temp_blue,temp_green,temp_red = cv2.split(temp_image)

    temp_blue = (temp_blue * blue_ratio)
    temp_green = (temp_green * green_ratio)
    temp_red = (temp_red * red_ratio)

    temp_blue = np.uint8(temp_blue)
    temp_green = np.uint8(temp_green)
    temp_red = np.uint8(temp_red)

    temp_result = cv2.merge((temp_blue,temp_green,temp_red))

    result = cv2.add(temp_result,image)

    #TODO RATIO OF AVERAGE FFS
    # calculate the mean of the outline contour with the given matrix calculation stuff
    # get the outside/inside region ratio, put it in the array
    # remove the current shadow on the final image

    #TODO add to array and calculate inside of region


cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
