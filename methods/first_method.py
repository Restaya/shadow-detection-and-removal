import cv2

image = cv2.imread("../test_pictures/first_method/lssd803.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Image", image)

# converts the image to lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# splits the channels of the image
light_level, a_level, b_level = cv2.split(lab_image)

# calculates the mean of each channel in the LAB colour space
a_level_mean = cv2.mean(a_level)[0]
b_level_mean = cv2.mean(b_level)[0]

# calculates the mean and standard deviation of the light level channel
light_level_mean, light_level_stddev = cv2.meanStdDev(light_level)

shadow_mask = lab_image.copy()
shadow_mask.fill(0)

# using morphological opening to erase smaller non-shadow pixels
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

if a_level_mean + b_level_mean <= 256:
    shadow_mask[light_level <= (light_level_mean - (light_level_stddev / 3))] = 255

    mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_RECT, struct)

# TODO according to the 2012 version
else:
    shadow_mask[light_level < light_level_stddev] = 255

    dst = cv2.dilate(shadow_mask, struct)
    dst2 = cv2.erode(dst, struct, iterations=2)
    mask = cv2.dilate(dst2, struct)


lab_image = cv2.cvtColor(lab_image,cv2.COLOR_LAB2BGR)
lab_image[mask > 0] = 255

cv2.imshow("Shadow detection",mask)
cv2.imshow("Result", lab_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
