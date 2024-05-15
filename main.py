import cv2
import os

from shadow_detection import *
from shadow_removal import *
from metrics import *

if __name__ == "__main__":

    # path of the image
    file_image = "images/test1.jpg"
    image = cv2.imread(file_image, cv2.IMREAD_COLOR)

    cv2.imshow("Original Image", cv2.imread(file_image))

    # select your choice of shadow detection you want to use by uncommenting the line

    e1 = cv2.getTickCount()

    # shadow_mask = cv2.imread("shadow_masks/test1.png",cv2.IMREAD_GRAYSCALE)
    shadow_mask = first_detection(file_image, False)
    # shadow_mask = second_detection(file_image, False)

    e2 = cv2.getTickCount()
    time1 = round((e2 - e1) / cv2.getTickFrequency(), 4)

    print("Shadow detection completed in: " + str(time1) + " seconds")

    # select your choice of shadow removal you want to use by uncommenting the line

    e3 = cv2.getTickCount()

    # first_removal(file_image, shadow_mask, None, False)
    # second_removal(file_image, shadow_mask, "inpainting")

    e4 = cv2.getTickCount()
    time2 = round((e4 - e3) / cv2.getTickFrequency(), 4)

    print("Shadow removal completed in: " + str(time2) + " seconds")
    print("------------------------------------------------------------------")

    # the results are saved in the results folder named respectively

    image_file_name = str.split((str.split(file_image, "/")[-1]), ".")[0]

    # mse calculation for shadow binary masks
    if os.path.exists("shadow_masks/" + image_file_name + ".png") or os.path.exists(
            "shadow_masks/" + image_file_name + ".jpg"):

        if os.path.exists("shadow_masks/" + image_file_name + ".png"):
            gt_mask_file = "shadow_masks/" + image_file_name + ".png"
        else:
            gt_mask_file = "shadow_masks/" + image_file_name + ".jpg"

        gt_mask = cv2.imread(gt_mask_file, cv2.IMREAD_GRAYSCALE)

        mse = mean_square_error(shadow_mask, gt_mask)

        print("The mean square error of the shadow detection is: " + str(mse))

    # mse calculation for images
    if os.path.exists("ground_truth_images/" + image_file_name + ".png") or os.path.exists("ground_truth_images/" + image_file_name + ".jpg"):

        if os.path.exists("ground_truth_images/" + image_file_name + ".png"):
            gt_image_file = "ground_truth_images/" + image_file_name + ".png"
        else:
            gt_image_file = "ground_truth_images/" + image_file_name + ".jpg"

        gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)

        mse = mean_square_error(image, gt_image)

        print("The mean square error of the image is: " + str(mse))

    # psnr calculation
    if os.path.exists("ground_truth_images/" + image_file_name + ".png") or os.path.exists("ground_truth_images/" + image_file_name + ".jpg"):

        if os.path.exists("ground_truth/" + image_file_name + ".png"):
            gt_image_file = "ground_truth/" + image_file_name + ".png"
        else:
            gt_image_file = "ground_truth/" + image_file_name + ".jpg"

        gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
        image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)

        psnr = cv2.PSNR(image, gt_image, 255)

        print("The peak signal to noise ratio is: " + psnr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
