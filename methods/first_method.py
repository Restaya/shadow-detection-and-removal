import cv2
import numpy as np

image = cv2.imread("../test_pictures/lssd229.jpg", cv2.IMREAD_COLOR)
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
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# creating external contours
contours_external, hierarchy_external = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

#TODO delete when properly finished
print("Test if the two contour arrays are equal\n")
print(len(corrected_contours) == len(corrected_external_contours))
print(str(len(corrected_contours)) + " " + str(len(corrected_external_contours)))

# based on the original mask, creating a better one to fill in smaller gaps using the found contours
contour_mask = np.zeros(image_shape, np.uint8)

# inside shadow region masks
inside_shadow_masks = []

# external contours masks
external_contour_masks = []

# creating a better mask with contours
sc_mask = np.zeros(image_shape, np.uint8)

#TODO fix for multiple contours, currently works properly, when their is only one
# test on lssd203.jpg, if it works
for i in range(len(corrected_contours)):

    temp_shadow_region_mask = np.zeros(image_shape, np.uint8)
    temp_external_contour_mask = np.zeros(image_shape, np.uint8)

    cv2.drawContours(sc_mask,corrected_contours,i,255,-1)

    # calculating the mean color values inside the shadow region
    cv2.drawContours(temp_shadow_region_mask,corrected_contours,i,255,-1)
    temp_shadow_mean = cv2.mean(image,temp_shadow_region_mask)
    inside_shadow_masks.append(temp_shadow_region_mask)

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

    blue, green, red = cv2.split(temp_image)

    blue = np.dot(blue, blue_ratio)
    green = np.dot(green,green_ratio)
    red = np.dot(red,red_ratio)

    blue = cv2.normalize(blue,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    green = cv2.normalize(green, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    red = cv2.normalize(red, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)

    temp_image = cv2.merge((blue,green,red))

    result = cv2.add(image,temp_image)


cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
