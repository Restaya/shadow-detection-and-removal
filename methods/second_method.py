import cv2
import numpy as np


def second_method(filename):
    e1 = cv2.getTickCount()

    image = cv2.imread("../test_pictures/" + filename + ".jpg", cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    cv2.imshow("Original", image)

    blue, green, red = cv2.split(image)

    # converting the image to grayscale with formula (1)
    gray_image = np.log(blue * (np.max(blue)/(np.min(blue)+1)) + green * (np.max(green)/(np.min(green)+1)) + red * (np.max(red)/(np.min(red)+1)))
    gray_image = cv2.normalize(gray_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Gray Image", gray_image)

    # start of watershed algorithm preprocessing
    thresh,gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # switch to one liner opening and closing
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.dilate(gray_image, struct)
    dst2 = cv2.erode(dst, struct, iterations=2)
    morp_image = cv2.dilate(dst2, struct)

    cv2.imshow("Preprocessed Gray Image",morp_image)



    # time it took to complete the method
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()

    print("Completed in : " + str(round(time, 4)) + " seconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    second_method("lssd9")
