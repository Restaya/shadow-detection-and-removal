import cv2
import numpy as np

from skimage.segmentation import watershed


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
        shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3) & (b_level <= b_level_mean - b_level_stddev / 3)] = 255

    rough_shadow_mask = shadow_mask.copy()

    # using morphological opening and closing to erase smaller non-shadow pixels
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((7, 7)))

    # using median blur to smoothen rough edges
    shadow_mask = cv2.medianBlur(shadow_mask, 7)

    if partial_results:

        # cv2.imshow("Original", image)

        #cv2.imshow("LAB Image", lab_image)

        cv2.imshow("Shadow Mask before morphological cleaning", rough_shadow_mask)

    shadow_mask = mask_correction(shadow_mask)

    cv2.imshow("Shadow Mask from first shadow detection method", shadow_mask)

    cv2.imwrite("results/First method shadow detection result.png", shadow_mask)

    return shadow_mask


def second_detection(file, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    blue, green, red = cv2.split(image)

    # finding the maximum and minimums element wise of the image
    image_max = cv2.max(blue, cv2.max(green, red))
    image_min = cv2.min(blue, cv2.min(green, red))

    # avoid division with zero
    image_min = np.where(image_min == 0, 1, image_min)
    image_max = np.where(image_max == 0, 1, image_max)

    gray_image = cv2.log(image_max / image_min)

    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    sure_fg = cv2.normalize(sure_fg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # mark the region of unknown with zero
    markers = watershed(dist_transform, markers, watershed_line=True)
    borders = image.copy()

    # getting the segments through unique values from the watershed result
    markers_list = np.unique(markers)
    print("Number of segments: " + str(len(markers_list - 1)))

    # constants needed for later calculation
    m = 1.31
    n = 1.19
    k1 = 0.8
    k2 = 1.2

    initial_shadow_mask = np.zeros(image_shape, np.uint8)

    # binary mask where the non-shadows are stored
    color_mean_non_shadow_mask = np.zeros(image_shape, np.uint8)

    # creating lists to store each individual segment,
    # 0th element is the watershed borders, None is placed as the first element because it is not used
    initial_shadow_segments = [None]
    initial_non_shadow_segments = [None]

    for value in markers_list:

        # skipping the image where borders are shown
        if value == 0:
            continue

        # getting each segment as a binary mask
        segment_mask = np.zeros(image_shape, np.uint8)
        segment_mask[markers == value] = 255

        # calculating the mean values of the segment
        segment_mean = cv2.mean(image, segment_mask)

        image_segment = image.copy()
        image_segment[segment_mask != 255] = 0

        # creates binary image where only pixels greater than the mean remain
        segment_non_shadow_mask = cv2.inRange(image_segment, segment_mean[:3], (255, 255, 255))

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
        max_delta_vector = max(delta_vector)
        min_delta_vector = min(delta_vector)

        # if red is max and blue is min
        if max_delta_vector == delta_vector[0] and min_delta_vector == delta_vector[2]:
            x_subtracted = cv2.subtract(red, blue)

        # if red is max and green is min
        if max_delta_vector == delta_vector[0] and min_delta_vector == delta_vector[1]:
            x_subtracted = cv2.subtract(red, green)

        # if green is max and blue is min
        if max_delta_vector == delta_vector[1] and min_delta_vector == delta_vector[2]:
            x_subtracted = cv2.subtract(green, blue)

        # if green is max and red is min
        if max_delta_vector == delta_vector[1] and min_delta_vector == delta_vector[0]:
            x_subtracted = cv2.subtract(green, red)

        # if blue is max and red is min
        if max_delta_vector == delta_vector[2] and min_delta_vector == delta_vector[0]:
            x_subtracted = cv2.subtract(blue, red)

        # if blue is max and green is min
        if max_delta_vector == delta_vector[2] and min_delta_vector == delta_vector[1]:
            x_subtracted = cv2.subtract(blue, green)

        x_mean = cv2.mean(x_subtracted, x_subtracted)[0]

        # if the pixel value is lower than x_mean, it's a possible shadow
        initial_shadow_segment = cv2.inRange(x_subtracted, 0, x_mean)
        initial_non_shadow_segment = cv2.inRange(x_subtracted, x_mean, 255)

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
        if value == 0:
            continue

        segment_mean_shadow = cv2.mean(image, initial_shadow_segments[value])
        segment_mean_non_shadow = cv2.mean(image, initial_non_shadow_segments[value])

        l_vector_first_value = m * (segment_mean_non_shadow[2] / segment_mean_non_shadow[0])
        l_vector_second_value = n * (segment_mean_non_shadow[1] / segment_mean_non_shadow[0])

        l_vector = (l_vector_first_value * delta_b, l_vector_second_value * delta_b, 1 * delta_b)
        l_interval = ((k1 * l_vector[0], k2 * l_vector[0]), (k1 * l_vector[1], k2 * l_vector[1]), (k1 * l_vector[2], k2 * l_vector[2]))

        blue_diff = segment_mean_non_shadow[0] - segment_mean_shadow[0]
        green_diff = segment_mean_non_shadow[1] - segment_mean_shadow[1]
        red_diff = segment_mean_non_shadow[2] - segment_mean_shadow[2]

        if (l_interval[0][0] < red_diff < l_interval[0][1]) and (l_interval[1][0] < green_diff < l_interval[1][1]) and (l_interval[2][0] < blue_diff < l_interval[2][1]):
            rough_shadow_mask[initial_shadow_segments[value] == 255] = 255

    # where the pixel are lower than the region mean
    color_mean_shadow_mask = cv2.bitwise_not(color_mean_non_shadow_mask)

    # based on the detected shadows and mean values, it creates the final shadow mask
    shadow_mask = cv2.bitwise_and(rough_shadow_mask, color_mean_shadow_mask)

    shadow_mask_before_cleaning = shadow_mask.copy()

    # morphological cleaning
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((5, 5)))

    shadow_mask = cv2.medianBlur(shadow_mask, 7)

    if partial_results:

        #cv2.imshow("Original", image)

        # cv2.imshow("Gray Image", gray_image)

        borders[markers == 0] = (0, 0, 255)
        cv2.imshow("Borders", borders)

        # cv2.imshow("Color mean shadow mask", color_mean_shadow_mask)

        # the initial detected shadow mask, the borders are from the watershed's algorithm segment borders
        cv2.imshow("Initial Shadow Mask", initial_shadow_mask)

        # the rough shadow mask, where non-shadow segments are removed
        #cv2.imshow("Rough Shadow Mask", rough_shadow_mask)

        # The shadow mask before morphological cleaning
        cv2.imshow("Shadow mask result before morphological cleaning", shadow_mask_before_cleaning)

        # showing the detected shadows on the original image
        shadow_image = image.copy()
        shadow_image[shadow_mask == 255] = 255

        # showing the watershed borders for visuals
        shadow_image[markers == -1] = [0, 0, 255]
        #cv2.imshow("Detected shadows with watershed borders", shadow_image)

    shadow_mask = mask_correction(shadow_mask)

    cv2.imshow("Shadow Mask from second shadow method", shadow_mask)

    cv2.imwrite("results/Second method shadow detection result.png", shadow_mask)

    return shadow_mask


def mask_correction(shadow_mask):

    _, shadow_segments = cv2.connectedComponents(shadow_mask, connectivity=8)
    shadow_segments_list = np.unique(shadow_segments)

    for value in shadow_segments_list:

        if value == 0:
            continue

        segment_mask = np.zeros(shadow_mask.shape, np.uint8)

        segment_mask[shadow_segments == value] = 1

        if np.count_nonzero(segment_mask) < 500:
            shadow_mask[shadow_segments == value] = 0

    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(shadow_mask.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, -1, 255, -1)

    mask = cv2.medianBlur(mask, 7)

    return mask
