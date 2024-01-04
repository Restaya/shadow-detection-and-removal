import cv2
import numpy as np


def first_detection(file, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)

    image_shape = image.shape[:2]

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

    # based on a threshold it detects the shadows on the image
    if a_level_mean + b_level_mean <= 256:
        shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3)] = 255
    else:
        shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3) & (
                    b_level <= b_level_mean - b_level_stddev / 3) & (
                    b_level >= -1 * b_level_mean + b_level_stddev / 3)] = 255

    rough_shadow_mask = shadow_mask.copy()

    # using morphological opening and closing to erase smaller non-shadow pixels
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((7, 7)))

    # using median blur to smoothen rough edges
    shadow_mask = cv2.medianBlur(shadow_mask, 7)

    cv2.imwrite("results/shadow_mask.png", shadow_mask)

    cv2.imshow("Shadow Mask", shadow_mask)

    print("Shadows detected, shadow mask saved in the results folder")

    if partial_results:

        # cv2.imshow("Original", image)

        # cv2.imshow("LAB Image", lab_image)

        cv2.imshow("Shadow Mask before morphological cleaning", rough_shadow_mask)

    return shadow_mask


def second_detection(file, partial_results=False):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    # blue, green, red = cv2.split(image)
    #     #
    #     # # finding the maximum and minimums element wise of the image
    #     # image_max = cv2.max(blue, cv2.max(green, red))
    #     # image_min = cv2.min(blue, cv2.min(green, red))
    #     #
    #     # # avoid division with zero
    #     # image_min = np.where(image_min == 0, 1, image_min)
    #     # image_max = np.where(image_max == 0, 1, image_max)
    #     #
    #     # gray_image = cv2.log(image_max / image_min)
    #     #
    #     # # note: fix normalization, currently from 0 to 3
    #     # gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5)))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((7, 7)))

    # sure background area
    sure_bg = cv2.dilate(thresh, np.ones((3, 3)), iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    # getting the segments through unique values from the watershed result
    markers_list = np.unique(markers)
    print("Number of segments: " + str(len(markers_list)))
    print(markers_list)

    # note: k2 original value is 1.2
    # constants needed for later calculation
    m = 1.31
    n = 1.19
    k1 = 0.8
    k2 = 1.5

    initial_shadow_mask = np.zeros(image_shape, np.uint8)

    # binary mask where the non-shadows are stored
    color_mean_non_shadow_mask = np.zeros(image_shape, np.uint8)

    initial_shadow_segments = []
    initial_non_shadow_segments = []

    # The first value in markers list is the borders, None at 0 index makes it able to not use value-1 at indexing
    initial_shadow_segments.append(None)
    initial_non_shadow_segments.append(None)

    for value in markers_list:

        # skipping the image where borders are shown
        if value == -1:
            continue

        # getting each segment as a binary mask
        segment_mask = np.zeros(image_shape, np.uint8)
        segment_mask[markers == value] = 255

        # calculating the mean values of the segment
        segment_mean = cv2.mean(image, segment_mask)

        image_segment = image.copy()
        image_segment[segment_mask != 255] = 0

        # creates binary image where only pixels greater than the mean remain
        segment_non_shadow_mask = cv2.inRange(image_segment, segment_mean[0], (255, 255, 255))

        # adds all non shadows to a binary image used for the final step to get the correct shadow boundaries
        color_mean_non_shadow_mask[segment_non_shadow_mask == 255] = 255

        # calculating the mean values in the non-shadow segment
        segment_mean_non_shadow = cv2.mean(image, segment_non_shadow_mask)

        # calculating the delta vectors values
        first_value = m * (segment_mean_non_shadow[2]/segment_mean_non_shadow[0])
        second_value = n * (segment_mean_non_shadow[1]/segment_mean_non_shadow[0])

        delta_vector = (first_value, second_value, 1)

        # getting the segment of the image
        initial_shadow_segment = image.copy()
        initial_shadow_segment[segment_mask != 255] = 0

        blue, green, red = cv2.split(initial_shadow_segment)

        # subtracting the minimal from the maximum based on the delta vector's two value
        if delta_vector[0] > delta_vector[1]:
            x_subtracted = cv2.subtract(red, blue)
        else:
            x_subtracted = cv2.subtract(green, blue)

        x_mean = cv2.mean(x_subtracted)

        # if the pixel value is lower than x_mean, it's a possible shadow
        initial_shadow_segment = cv2.inRange(x_subtracted, 0, x_mean[0])
        initial_non_shadow_segment = cv2.inRange(x_subtracted, x_mean[0], 255)

        # for visual of the initial shadow mask
        initial_shadow_segment[segment_mask == 0] = 0

        # adding both non-shadow and shadow segments to a list for later calculation
        initial_shadow_segments.append(initial_shadow_segment)
        initial_non_shadow_segments.append(initial_non_shadow_segment)

        initial_shadow_mask[initial_shadow_segment == 255] = 255

    initial_non_shadow_mask = cv2.bitwise_not(initial_shadow_mask)

    delta_b = -1 * (cv2.mean(image, initial_shadow_mask)[0] - cv2.mean(image, initial_non_shadow_mask)[0])

    rough_shadow_mask = np.zeros(image_shape, np.uint8)

    for value in markers_list:

        # skipping the image where borders are shown
        if value == -1:
            continue

        segment_mean_shadow = cv2.mean(image, initial_shadow_segments[value])
        segment_mean_non_shadow = cv2.mean(image, initial_non_shadow_segments[value])

        l_vector_first_value = m * (segment_mean_non_shadow[2]/segment_mean_non_shadow[0])
        l_vector_second_value = n * (segment_mean_non_shadow[1]/segment_mean_non_shadow[0])

        l_vector = (l_vector_first_value * delta_b, l_vector_second_value * delta_b, 1 * delta_b)
        l_interval = ((k1 * l_vector[0], k2 * l_vector[0]), (k1 * l_vector[1], k2 * l_vector[1]), (k1 * l_vector[2], k2 * l_vector[2]))

        blue_diff = segment_mean_non_shadow[0] - segment_mean_shadow[0]
        green_diff = segment_mean_non_shadow[1] - segment_mean_shadow[1]
        red_diff = segment_mean_non_shadow[2] - segment_mean_shadow[2]

        if (l_interval[0][0] < red_diff < l_interval[0][1]) and (l_interval[1][0] < green_diff < l_interval[1][1]) and (l_interval[2][0] < blue_diff < l_interval[2][1]):
            rough_shadow_mask[initial_shadow_segments[value] == 255] = 255

        # cv2.imshow(str(value), initial_non_shadow_segments[value])

    # where the pixel are lower than the region mean
    color_mean_shadow_mask = cv2.bitwise_not(color_mean_non_shadow_mask)

    # based on the detected shadows and mean values, it creates the final shadow mask
    shadow_mask = cv2.bitwise_and(rough_shadow_mask, color_mean_shadow_mask)

    cv2.imwrite("../results/shadow_mask.png", shadow_mask)

    cv2.imshow("Shadow Mask", shadow_mask)

    if partial_results:

        # cv2.imshow("Original", image)

        # cv2.imshow("Gray Image", gray_image)

        # showing the watershed borders
        water_shed_image = image.copy()
        water_shed_image[markers == -1] = [0, 0, 255]
        cv2.imshow("Watershed Borders", water_shed_image)

        # cv2.imshow("Color mean shadow mask", color_mean_shadow_mask)

        # the initial detected shadow mask, the borders are from the watershed's algorithm segment borders
        cv2.imshow("Initial Shadow Mask", initial_shadow_mask)

        # the rough shadow mask, where non-shadow segments are removed
        # cv2.imshow("Rough Shadow Mask", rough_shadow_mask)

        # showing the detected shadows on the original image
        shadow_image = image.copy()
        shadow_image[shadow_mask == 255] = 255

        # showing the watershed borders for visuals
        shadow_image[markers == -1] = [0, 0, 255]
        cv2.imshow("Detected shadows with watershed borders", shadow_image)

    print("Shadows detected, shadow mask saved in the results folder")

    return shadow_mask
