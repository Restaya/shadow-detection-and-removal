import cv2
import numpy as np


def first_method(filename):

    e1 = cv2.getTickCount()

    image = cv2.imread("../test_pictures/" + filename + ".jpg", cv2.IMREAD_COLOR)
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

    # based on a threshold it detects the shadows on the image
    if a_level_mean + b_level_mean <= 256:
        shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3)] = 255
    else:
        shadow_mask[(light_level <= light_level_mean - light_level_stddev / 3) & (
                    b_level <= b_level_mean - b_level_stddev / 3) & (
                                b_level >= -1 * b_level_mean + b_level_stddev / 3)] = 255

    # using morphological opening to erase smaller non-shadow pixels
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    dst = cv2.dilate(shadow_mask, struct)
    dst2 = cv2.erode(dst, struct, iterations=2)
    mask = cv2.dilate(dst2, struct)

    # using median blur to smoothen rough edges
    mask = cv2.medianBlur(mask, 3)

    # finding contours with mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # the arc threshold isn't accurate, since it's not specified in the paper
    # filtering out the smaller non-shadow contours
    corrected_contours = []
    for contour in contours:
        if cv2.arcLength(contour, True) > 60:
            corrected_contours.append(contour)

    # contains the edges of the shadow regions
    shadow_region_edges = np.zeros(image_shape, np.uint8)

    for i in range(len(corrected_contours)):
        # temporary mask for each shadow region
        temp_shadow_region_mask = np.zeros(image_shape, np.uint8)

        # drawing the individual contours
        cv2.drawContours(temp_shadow_region_mask, corrected_contours, i, 255, -1)

        # dilates the image to get a bigger shadow region
        dil_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        external_region_mask = cv2.dilate(temp_shadow_region_mask, dil_struct)

        # creating the mask of just outside the shadow region
        external_region_mask_contour = cv2.subtract(external_region_mask, temp_shadow_region_mask)

        # creating a mask with only the outlines of the shadow region
        shadow_region_edges = cv2.add(external_region_mask_contour, shadow_region_edges)

        # calculatiog the insde and just outside means of the shadow region
        shadow_mean = cv2.mean(image, external_region_mask)
        outside_mean = cv2.mean(image, external_region_mask_contour)

        # calculating the ratio of outside to inside
        blue_ratio = round(outside_mean[0] / shadow_mean[0], 4)
        green_ratio = round(outside_mean[1] / shadow_mean[1], 4)
        red_ratio = round(outside_mean[2] / shadow_mean[2], 4)

        # copying the original image and zeroing where there's a shadow
        masked_image = image.copy()
        masked_image[external_region_mask != 255] = 0

        blue, green, red = cv2.split(masked_image)

        # multiplying the BGR channels with the constant ratios
        blue = np.dot(blue, blue_ratio)
        green = np.dot(green, green_ratio)
        red = np.dot(red, red_ratio)

        result_image = cv2.merge((blue, green, red))

        # converting the matrix from float to integer matrix
        result_image = cv2.normalize(result_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        result = cv2.add(image, result_image)

    # creating a mask for edge smoothing
    edge_mask = np.zeros(result.shape[:2], np.uint8)
    edge_mask[shadow_region_edges == 255] = 1

    edge_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_mask = cv2.dilate(edge_mask, edge_struct)

    edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

    # inverting the edge mask
    inverted_edge_mask = ~edge_mask + 2

    # gauss blur on result image
    result_gaussian = cv2.GaussianBlur(result, (7, 7), 2, 2)

    # with this equation the over illuminated edges are less bright
    result = (result_gaussian * edge_mask) + (result * inverted_edge_mask)

    cv2.imshow("Result", result)

    # time it took to complete the method
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()

    print("Completed in : " + str(round(time, 4)) + " seconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    first_method("lssd9")
