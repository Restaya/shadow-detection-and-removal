import cv2
import numpy as np


def second_method_detection(filename):
    image = cv2.imread("../test_pictures/" + filename + ".jpg", cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    # cv2.imshow("Original", image)

    blue, green, red = cv2.split(image)

    # finding the maximums and minimums element wise of the image
    image_max = np.maximum(blue, np.maximum(green, red))
    image_min = np.minimum(blue, np.minimum(green, red))

    # avoid division with zero
    image_min = np.where(image_min == 0, 1, image_min)
    image_max = np.where(image_max == 0, 1, image_max)

    gray_image = cv2.log(image_max / image_min)
    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Gray Image", gray_image)

    # preprocessing for watershed segmentation
    _, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cv2.imshow("Preprocessed Gray Image", gray_image)

    # note: NEED TO CHANGE, LOWER NUMBER OF SEGMENTS OR SOMETHING
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("Morphed Image",opening)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, markers = cv2.connectedComponents(dist_transform)

    markers = cv2.watershed(image, markers)

    # getting the segments through unique values from the watershed result
    markers_list = np.unique(markers)
    print("Number of segments:  " + str(len(markers_list)))

    # note : showing boundaries, for visuals
    watershed_borders = image.copy()
    watershed_borders[markers == -1] = [0, 0, 255]

    cv2.imshow("Watershed Result", watershed_borders)

    # constants needed for later calculation
    m = 1.31
    n = 1.19
    k1 = 0.8
    k2 = 1.2

    initial_shadow_mask = np.zeros(image_shape, np.uint8)

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
        segment_non_shadow_mask = cv2.inRange(image_segment, segment_mean, (255, 255, 255))

        # calculating the mean values in the non-shadow segment
        segment_mean_non_shadow = cv2.mean(image,segment_non_shadow_mask)

        # calculating the delta vectors values
        first_value = m * (segment_mean_non_shadow[2]/segment_mean_non_shadow[0])
        second_value = n * (segment_mean_non_shadow[1]/segment_mean_non_shadow[0])

        delta_vector = (first_value,second_value,1)

        # getting the segment of the image
        initial_shadow_segment = image.copy()
        initial_shadow_segment[segment_mask != 255] = 0

        blue, green, red = cv2.split(initial_shadow_segment)

        # subtracting the minimal from the maximum based on the delta vector's two value
        if delta_vector[0] > delta_vector[1]:
            x_subtracted = cv2.subtract(red,blue)
        else:
            x_subtracted = cv2.subtract(green,blue)

        x_mean = cv2.mean(x_subtracted)

        # if the pixel value is lower than x_mean, it's a possible shadow
        initial_shadow_segment = cv2.inRange(x_subtracted,0,x_mean[0])
        initial_non_shadow_segment = cv2.inRange(x_subtracted,x_mean[0], 255)

        # for visual of the initial shadow mask
        initial_shadow_segment[segment_mask == 0] = 0

        # adding both non-shadow and shadow segments to a list for later calculation
        initial_shadow_segments.append(initial_shadow_segment)
        initial_non_shadow_segments.append(initial_non_shadow_segment)

        initial_shadow_mask[initial_shadow_segment == 255] = 255
        #cv2.imshow(str(value), segment_mask)

    initial_non_shadow_mask = cv2.bitwise_not(initial_shadow_mask)

    delta_b = -1 * (cv2.mean(image, initial_shadow_mask)[0] - cv2.mean(image, initial_non_shadow_mask)[0])

    # note: IT'S STILL NOT WORKING PROPERLY, NEED FURTHER DEBUGGING

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

        #cv2.imshow(str(value), initial_non_shadow_segments[value])

    # the initial detected shadow mask, the borders are from the watershed's algorithm segment borders
    cv2.imshow("Initial Shadow Mask",initial_shadow_mask)

    # the rough shadow mask, where non-shadow segments are corrected
    #cv2.imshow("Rough Shadow Mask", rough_shadow_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    second_method_detection("lssd297")
