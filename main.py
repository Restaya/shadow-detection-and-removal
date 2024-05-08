from shadow_detection import *
from shadow_removal import *

import cv2

if __name__ == "__main__":

    # path of the image
    file_image = "images/lssd9.jpg"

    cv2.imshow("Original Image", cv2.imread(file_image))

    # select your choice of shadow detection you want to use by uncommenting the line

    e1 = cv2.getTickCount()

    #shadow_mask = cv2.imread("shadow_masks/93-1.png",cv2.IMREAD_GRAYSCALE)
    shadow_mask = first_detection(file_image,False)
    #shadow_mask = second_detection(file_image, True)

    e2 = cv2.getTickCount()
    time1 = round((e2 - e1) / cv2.getTickFrequency(), 4)

    print("Shadow detection completed in: " + str(time1) + " seconds")

    # select your choice of shadow removal you want to use by uncommenting the line

    e3 = cv2.getTickCount()

    #first_removal(file_image, shadow_mask)
    second_removal(file_image, shadow_mask, True)

    e4 = cv2.getTickCount()
    time2 = round((e4 - e3) / cv2.getTickFrequency(), 4)

    print("Shadow removal completed in: " + str(time2) + " seconds")

    # the results are saved in the results folder named respectively

    cv2.waitKey(0)
    cv2.destroyAllWindows()

