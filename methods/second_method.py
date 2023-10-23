import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

import multiprocessing


def second_method(filename):

    e1 = cv2.getTickCount()

    # gets the amount of cpu cores for parallel computing
    cpu_cores = multiprocessing.cpu_count()
    cpu_cores //= 2

    image = cv2.imread("../test_pictures/" + filename + ".jpg", cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]

    cv2.imshow("Original", image)

    image = cv2.medianBlur(image, 3)

    flat_image = image.reshape((-1, 3))
    flat_image = np.float32(flat_image)

    # mean shift
    bandwidth = estimate_bandwidth(flat_image, quantile=0.06, n_samples=1000, n_jobs=cpu_cores)
    mean_shift = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True, n_jobs=cpu_cores)
    mean_shift.fit(flat_image)
    labeled = mean_shift.labels_

    segments = np.unique(labeled)
    print('Number of segments: ' + str(segments.shape[0]))

    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total / count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((image.shape))

    # time it took to complete the method
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()

    cv2.imshow("Result", result)

    print("Completed in : " + str(round(time, 4)) + " seconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    second_method("lssd9")
