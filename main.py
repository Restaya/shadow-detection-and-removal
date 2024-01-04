from shadow_detection import *
from shadow_removal import *

import cv2

if __name__ == "__main__":

    # path of the image
    file_image = "test_pictures/lssd9.jpg"

    cv2.imshow("Original Image",cv2.imread(file_image))

    # select your choice of shadow detection you want to use by uncommenting the line

    shadow_mask = first_detection(file_image)
    #shadow_mask = second_detection(file_image)

    # select your choice of shadow removal you want to use by uncommenting the line

    first_removal(file_image, shadow_mask)

    # the results are saved in the results folder named respectively

    cv2.waitKey(0)
    cv2.destroyAllWindows()

