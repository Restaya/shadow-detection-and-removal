import cv2
import numpy as np

from shadow_detection import *
from shadow_removal import *

from skimage.restoration import inpaint

if __name__ == "__main__":

    file_image = "images/lssd9.jpg"

    shadow_mask = first_detection(file_image)

    shadow_removal_result = first_removal(file_image, shadow_mask)

    dilated_shadow_mask = cv2.dilate(shadow_mask, np.ones((7, 7)))

    edge = cv2.subtract(dilated_shadow_mask, shadow_mask)

    # cv2.imshow("Edge", edge)

    result = inpaint.inpaint_biharmonic(shadow_removal_result, edge, channel_axis=-1)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Result", result)

    cv2.waitKey()
    cv2.destroyAllWindows()
