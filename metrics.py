import numpy as np
import cv2
from math import log10,sqrt

# calculates the mean square error
def mean_square_error(image, gt_image):

    image_shape = image.shape[:2]
    gt_image_shape = gt_image.shape[:2]

    if image_shape[0] != gt_image_shape[0] or image_shape[1] != gt_image_shape[1]:
        print("The two input image size is not equal")
        return

    if len(image_shape) != len(gt_image_shape):
        print("Not equal channel numbers")
        return

    if len(image_shape) == 2:
        return round((np.mean((gt_image - image) ** 2)), 2)

    else:
        b, g, r = cv2.split(image)
        bt, gt, rt = cv2.split(gt_image)

        b_mse = np.mean((bt - b) ** 2)
        g_mse = np.mean((gt - g) ** 2)
        r_mse = np.mean((rt - r) ** 2)

        return round(((b_mse + g_mse + r_mse)/3) * 100, 2)


def peak_signal_to_noise_ratio(mse):

    if mse == 0:
        return 0

    return round(20 * log10(255 / sqrt(mse)), 2)





