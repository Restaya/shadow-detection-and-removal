import cv2
import numpy as np

from shadow_detection import *
from shadow_removal import *

if __name__ == "__main__":

    file_image = "test_images/lssd178.jpg"

    shadow_mask = first_detection(file_image)

    shadow_removal_result = second_removal(file_image, shadow_mask)

    dilated_shadow_mask = cv2.dilate(shadow_mask, np.ones((7, 7)))

    edge = cv2.subtract(dilated_shadow_mask, shadow_mask)

    # cv2.imshow("Edge", edge)

    blurred = cv2.medianBlur(shadow_removal_result, 5)

    blurred[edge != 255] = 0
    shadow_removal_result[edge == 255] = 0

    cv2.imshow("Median Blurred", blurred)

    result = cv2.add(shadow_removal_result,blurred)

    cv2.imshow("Result",result)

    cv2.waitKey()
    cv2.destroyAllWindows()
