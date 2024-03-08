import cv2
import numpy as np
import math

from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.exposure import match_histograms


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

    cv2.imwrite("results/shadow_free_first_method.png", result)

    cv2.imshow("Result from first shadow removal", result)

    print("Successful removal, image saved in the results folder")

    if partial_results:

        shadow_edge_mask[shadow_edge_mask == 1] = 255
        cv2.imshow("Shadow edges", shadow_edge_mask)

    return result


def second_removal(file, shadow_mask, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)

    image_shape = image.shape[:2]

    shadow_image = image.copy()
    non_shadow_image = image.copy()

    #shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_DILATE, np.ones((3, 3)))

    non_shadow_mask = cv2.bitwise_not(shadow_mask)
    #non_shadow_mask = cv2.morphologyEx(non_shadow_mask, cv2.MORPH_ERODE, np.ones((15, 15)))

    shadow_image[shadow_mask != 255] = 0
    non_shadow_image[shadow_mask == 255] = 0

    # note: not converting the shadow segments to lab seems to improve removal results
    # TODO test without mask, only try on zero'd out image, may improve
    # using the SLIC algorithm to segment the image in non-shadow and shadow regions
    shadow_segments = slic(image, n_segments=70, compactness=20, mask=shadow_mask, convert2lab=True, enforce_connectivity=True)
    non_shadow_segments = slic(image, n_segments=70, compactness=20, mask=non_shadow_mask, convert2lab=True, enforce_connectivity=True)

    # making a list out of the labels
    shadow_segments_list = np.unique(shadow_segments)
    non_shadow_segments_list = np.unique(non_shadow_segments)

    print("Number of shadow segments: " + str(len(shadow_segments_list)))
    print("Number of non-shadow segments: " + str(len(non_shadow_segments_list)))

    # list for storing the centroids of the shadow segments
    shadow_segments_centers = [None]

    shadow_segments_center = np.zeros(image.shape, np.uint8)

    # calculating the shadow segments centers
    for value in shadow_segments_list:

        # ignore the background label
        if value == 0:
            continue

        segment_mask = np.zeros(image_shape, np.uint8)
        segment_mask[shadow_segments == value] = 255

        moments = cv2.moments(segment_mask)

        # calculating the center coordinates of the segment
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # if the center is out of bounds, place it at the border of the image
        if cx > image_shape[1]:
            cx = image_shape[1]

        if cy > image_shape[0]:
            cy = image_shape[0]

        # storing the x and y coordinates of the center
        shadow_segments_centers.append((cx, cy))

        # drawing a circle on the image to showcase the centers
        cv2.circle(shadow_segments_center, (cx, cy), 3, (0, 0, 255), -1)

    # showing the SLIC segmentations result in shadow area
    shadow_segments_colored = label2rgb(shadow_segments, shadow_image, kind="avg")

    # showing the centroids of the shadow segments
    shadow_segments_colored_centroids = cv2.add(shadow_segments_colored, shadow_segments_center)

    # list for storing the centroids of the non-shadow segments
    non_shadow_segments_centers = [None]

    non_shadow_segments_center = np.zeros(image.shape, np.uint8)

    # calculating the non-shadow segments centers
    for value in non_shadow_segments_list:

        # ignore the background label
        if value == 0:
            continue

        segment_mask = np.zeros(image_shape, np.uint8)
        segment_mask[non_shadow_segments == value] = 255

        moments = cv2.moments(segment_mask)

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # if the center is out of bounds, place it at the border of the image
        if cx > image_shape[1]:
            cx = image_shape[1]

        if cy > image_shape[0]:
            cy = image_shape[0]

        # storing the x and y coordinates of the center
        non_shadow_segments_centers.append((cx, cy))

        # drawing a circle on the image to showcase the centroids
        cv2.circle(non_shadow_segments_center, (cx, cy), 3, (0, 255, 255), -1)

    # showing the SLIC segmentations result in non-shadow area
    non_shadow_segments_colored = label2rgb(non_shadow_segments, non_shadow_image, kind="avg")

    # showing the centroids of the non-shadow segments
    non_shadow_segments_colored_centroids = cv2.add(non_shadow_segments_colored, non_shadow_segments_center)

    # adding the two center containing images together to showcase who is paired with who
    image_segment_centers = cv2.add(non_shadow_segments_colored_centroids, shadow_segments_colored_centroids)

    # image showing lines between the optimal pairs
    image_segment_centers_lines = image_segment_centers.copy()

    # storing which non-shadow segment is closest to the shadow segment
    label_pairs = [None]

    # calculating the possible max Euclidean distance to initialize
    max_distance = math.dist((0, 0), image_shape)

    for s_value in shadow_segments_list:

        if s_value == 0:
            continue

        distance = max_distance
        closest_label = None

        shadow_center = shadow_segments_centers[s_value]

        for n_value in non_shadow_segments_list:

            if n_value == 0:
                continue

            non_shadow_center = non_shadow_segments_centers[n_value]

            possible_distance = math.dist(shadow_center, non_shadow_center)

            if possible_distance < distance:
                distance = possible_distance
                closest_label = n_value
                final_non_shadow_center = non_shadow_center

        # index is the shadow segment label, the element is the non-shadow segment label
        label_pairs.append(closest_label)

        # draws a line between the two centers
        cv2.line(image_segment_centers_lines, shadow_center, final_non_shadow_center, (255, 0, 0), 1)

    result = non_shadow_image.copy()

    # relighting the shadow with the optimal non-shadow pair
    for i in range(len(label_pairs)):

        if i == 0:
            continue

        # index is the label value of the shadow segment
        # the element is the label value of the non-shadow segment

        # empty masks for the segments
        shadow_segment = image.copy()
        non_shadow_segment = image.copy()

        shadow_segment[shadow_segments != i] = 0
        non_shadow_segment[non_shadow_segments != label_pairs[i]] = 0

        relighted_segment = match_histograms(shadow_segment, non_shadow_segment, channel_axis=-1)
        relighted_segment[shadow_segments != i] = 0

        result = cv2.add(result, relighted_segment)

    cv2.imshow("Result from second shadow removal method", result)

    cv2.imwrite("results/shadow_free_second_method.png", result)

    if partial_results:

        # showing the SLIC segmentations result in shadow area
        #cv2.imshow("Shadow segments", shadow_segments_colored)

        # showing the SLIC segmentations result in non-shadow area
        #cv2.imshow("Non-Shadow segments", non_shadow_segments_colored)

        # showing the centroids of the shadow segments
        #cv2.imshow("Centers of shadow segments", shadow_segments_colored_centroids)

        # showing the centroids of the non-shadow segments
        #cv2.imshow("Centers of non-shadow segments", non_shadow_segments_colored_centroids)

        # showing the centroids in one image
        #cv2.imshow("Centers of the segments", image_segment_centers)

        # showing the optimal pairs chosen with lines drawn between the centers
        cv2.imshow("Lines between the pairs", image_segment_centers_lines)





