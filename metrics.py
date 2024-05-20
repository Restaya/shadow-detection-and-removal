import numpy as np
import cv2
from math import log10, sqrt


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

    return round(np.mean((gt_image - image) ** 2), 4)


def peak_signal_to_noise_ratio(mse):

    if mse == 0:
        print("The selected input image was the ground truth image itself")
        return 0

    return round(20 * log10(255 / sqrt(mse)), 2)
