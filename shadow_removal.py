import cv2
import numpy as np


def first_removal(file, shadow_mask, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)

    image_shape = image.shape[:2]

    # labeling all white segments
    _, markers = cv2.connectedComponents(shadow_mask, connectivity=8)

    markers_list = np.unique(markers)

    # stores the actual values of shadow segments
    corrected_markers_list = []

    # removing small non shadows based on pixel size
    for value in markers_list:

        if value == 0:
            continue

        temp_mask = np.zeros(image_shape, np.uint8)
        temp_mask[markers == value] = 255

        # the used threshold value is based on testing, the paper doesn't include it
        if cv2.countNonZero(temp_mask) > 200:
            corrected_markers_list.append(value)

    print("Number of shadow segments: " + str(len(corrected_markers_list)))

    # storing the shadow edges
    shadow_edge_mask = np.zeros(image_shape, np.uint8)

    for value in corrected_markers_list:

        shadow_segment = np.zeros(image_shape, np.uint8)
        shadow_segment[markers == value] = 255

        # dilating the segment
        dilated_shadow_segment = cv2.dilate(shadow_segment, np.ones((5, 5)))

        # creating a mask where only the edge remains of the segment
        edge_shadow_segment = cv2.subtract(dilated_shadow_segment, shadow_segment)
        shadow_edge_mask[edge_shadow_segment == 255] = 255

        shadow_segment_mean = cv2.mean(image, shadow_segment)
        edge_shadow_segment_mean = cv2.mean(image, edge_shadow_segment)

        # calculating the just outside to inside ratio
        blue_ratio = round(edge_shadow_segment_mean[0] / shadow_segment_mean[0], 4)
        green_ratio = round(edge_shadow_segment_mean[1] / shadow_segment_mean[1], 4)
        red_ratio = round(edge_shadow_segment_mean[2] / shadow_segment_mean[2], 4)

        # copying the original image and zeroing where there's a shadow
        mask_image = image.copy()
        mask_image[dilated_shadow_segment != 255] = 0

        blue, green, red = cv2.split(mask_image)

        # multiplying the BGR channels with the constant ratios
        blue = np.dot(blue, blue_ratio)
        green = np.dot(green, green_ratio)
        red = np.dot(red, red_ratio)

        result_image = cv2.merge((blue, green, red))

        # converting the matrix from float to integer matrix
        result_image = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        result = cv2.add(image, result_image)

    # switching the 255 to 1 for matrix calculation
    shadow_edge_mask[shadow_edge_mask == 255] = 1

    edge_mask = cv2.dilate(shadow_edge_mask, np.ones((5, 5)))

    edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

    # inverting the edge mask
    inverted_edge_mask = ~edge_mask + 2

    # gauss blur on result image
    result_gaussian = cv2.GaussianBlur(result, (5, 5), 2, 2)

    # with this equation the over illuminated edges are less bright
    result = (result_gaussian * edge_mask) + (result * inverted_edge_mask)

    cv2.imwrite("results/shadow_free.png", result)

    cv2.imshow("Result", result)

    print("Successful removal, image saved in the results folder")

    if partial_results:

        shadow_edge_mask[shadow_edge_mask == 1] = 255
        cv2.imshow("Shadow edges", shadow_edge_mask)

    return result
