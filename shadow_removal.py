import cv2
import numpy as np
import math

from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.exposure import match_histograms
from skimage.restoration import inpaint


def first_removal(file, shadow_mask, post_processing_operation=None, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)

    image_shape = image.shape[:2]

    # labeling all white segments
    _, markers = cv2.connectedComponents(shadow_mask, connectivity=8)

    markers_list = np.unique(markers)

    print("Number of shadow segments: " + str(len(markers_list) - 1))

    # image to store all relighted segments
    relighted_shadow_segments = np.zeros(image.shape, np.uint8)

    # storing the shadow edges
    shadow_edge_mask = np.zeros(image_shape, np.uint8)

    for value in markers_list:

        if value == 0:
            continue

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
        mask_image[shadow_segment != 255] = 0

        blue, green, red = cv2.split(mask_image)

        # multiplying the BGR channels with the constant ratios
        blue = np.dot(blue, blue_ratio)
        green = np.dot(green, green_ratio)
        red = np.dot(red, red_ratio)

        result_image = cv2.merge((blue, green, red))

        # converting the matrix from float to integer matrix
        result_image = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        relighted_shadow_segments = cv2.add(relighted_shadow_segments, result_image)

    result = cv2.add(image, relighted_shadow_segments)

    if post_processing_operation is not None:

        result = post_processing(result, shadow_mask, post_processing_operation)

    cv2.imwrite("results/First method shadow removal result.png", result)

    cv2.imshow("Result from first shadow removal", result)

    if partial_results:

        shadow_edge_mask[shadow_edge_mask == 1] = 255
        cv2.imshow("Shadow edges", shadow_edge_mask)

    return result


def second_removal(file, shadow_mask, post_processing_operation=None, partial_results=False):

    image = cv2.imread(file, cv2.IMREAD_COLOR)

    image_shape = image.shape[:2]

    shadow_image = image.copy()
    non_shadow_image = image.copy()

    non_shadow_mask = cv2.bitwise_not(shadow_mask)

    shadow_image[shadow_mask != 255] = 0
    non_shadow_image[shadow_mask == 255] = 0

    # using the SLIC algorithm to segment the image in non-shadow and shadow regions
    shadow_segments = slic(image, n_segments=70, compactness=10, mask=shadow_mask)
    non_shadow_segments = slic(image, n_segments=60, compactness=8, mask=non_shadow_mask)

    # making a list out of the labels
    shadow_segments_list = np.unique(shadow_segments)
    non_shadow_segments_list = np.unique(non_shadow_segments)

    print("Number of shadow segments: " + str(len(shadow_segments_list) - 1))
    print("Number of non-shadow segments: " + str(len(non_shadow_segments_list) - 1))

    # list for storing the centroids of the shadow segments
    shadow_segments_centers = [None]

    shadow_segments_center = np.zeros(image.shape, np.uint8)

    # calculating the shadow segments centers
    for value in shadow_segments_list:

        # ignore the background label
        if value == 0:
            continue

        # creating a mask for the current segment
        segment_mask = np.zeros(image_shape, np.uint8)
        segment_mask[shadow_segments == value] = 255

        # used to calculate the center of the segment
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

    # finding the best pair for each shadow segment based on minimal Euclidean distance
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

            # calculating the Euclidean distance between two points
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

    if post_processing_operation is not None:

        result = post_processing(result, shadow_mask, post_processing_operation)

    cv2.imshow("Result from second shadow removal method", result)

    cv2.imwrite("results/Second method shadow removal result.png", result)

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

    return result


def post_processing(image, shadow_mask, operation):

    dilated_shadow_mask = cv2.dilate(shadow_mask, np.ones((7, 7)))
    edge = cv2.subtract(dilated_shadow_mask, shadow_mask)

    if operation == "median":

        blurred = cv2.medianBlur(image, 5)

        blurred[edge != 255] = 0
        image[edge == 255] = 0

        result = cv2.add(image, blurred)

        print("Used median blurr for post processing!")

    if operation == "inpainting":

        result = inpaint.inpaint_biharmonic(image, edge, channel_axis=-1)

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        print("Used inpainting for post processing!")

    return result