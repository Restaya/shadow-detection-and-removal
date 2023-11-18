import cv2
import numpy as np


def second_method_detection(filename):

    image = cv2.imread("../test_pictures/" + filename + ".jpg", cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    #cv2.imshow("Original", image)

    blue, green, red = cv2.split(image)

    # converting the image to grayscale with formula (1)
    gray_image = cv2.log(blue * (np.max(blue)/(np.min(blue)+1)) + green * (np.max(green)/(np.min(green)+1)) + red * (np.max(red)/(np.min(red)+1)))
    gray_image = cv2.normalize(gray_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    #cv2.imshow("Gray Image", gray_image)

    # preprocessing for watershed segmentation
    thresh,gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cv2.imshow("Preprocessed Gray Image", gray_image)

    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("Morphed Image",opening)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _,markers = cv2.connectedComponents(dist_transform)

    markers = cv2.watershed(image,markers)

    # getting the segments through unique values from the watershed result
    markers_list = np.unique(markers)
    print("Number of segments:  " + str(len(markers_list)))

    # note : showing boundaries, for visuals
    watershed_borders = image.copy()
    watershed_borders[markers == -1] = [0, 0, 255]

    cv2.imshow("Watershed Result", watershed_borders)

    for value in markers_list:

        if value == -1:
            continue

        region_mask = np.zeros(image_shape, np.uint8)
        region_mask[markers == value] = 255

        #cv2.imshow(str(value) + "th Region mask",region_mask)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    second_method_detection("lssd24")
