import os
from PyQt5.QtWidgets import *
from PyQt5 import uic

from shadow_detection import *
from shadow_removal import *
from metrics import *


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # loading the UI file
        uic.loadUi("menu.ui", self)

        # defining the buttons
        self.button_choose_image = self.findChild(QPushButton, "push_button_choose_image")
        self.button_choose_mask = self.findChild(QPushButton, "push_button_choose_mask")

        self.button_show_image = self.findChild(QPushButton, "push_button_show_image")
        self.button_show_mask = self.findChild(QPushButton, "push_button_show_mask")

        self.button_detect_shadows = self.findChild(QPushButton, "push_button_detect_shadows")
        self.button_remove_shadows = self.findChild(QPushButton, "push_button_remove_shadows")

        self.label_file_name = self.findChild(QLabel, "label_file_name")
        self.label_mask_name = self.findChild(QLabel, "label_mask_name")

        # defining radio buttons
        self.radio_button_first_detection = self.findChild(QRadioButton, "radio_button_first_detection")
        self.radio_button_second_detection = self.findChild(QRadioButton, "radio_button_second_detection")

        self.radio_button_first_removal = self.findChild(QRadioButton, "radio_button_first_removal")
        self.radio_button_second_removal = self.findChild(QRadioButton, "radio_button_second_removal")

        # defining postprocessing radio buttons
        self.radioButton_no_pp = self.findChild(QRadioButton, "radioButton_no_pp")
        self.radioButton_inpainting_pp = self.findChild(QRadioButton, "radioButton_inpainting_pp")
        self.radioButton_median_blur_pp = self.findChild(QRadioButton, "radioButton_median_blur_pp")

        self.check_box_partial_results = self.findChild(QCheckBox, "check_box_partial_results")
        self.partial_results = False

        self.image_path = None
        self.image_file_name = None
        self.mask_path = None
        self.shadow_mask = None
        self.gt_mask = None
        self.gt_image = None
        self.post_processing_operation = None

        # defining the choose image button
        self.button_choose_image.clicked.connect(self.choose_image)

        # defining the choose shadow mask button
        self.button_choose_mask.clicked.connect(self.choose_mask)

        # defining the image showing button
        self.button_show_image.clicked.connect(self.show_image)

        # defining the shadow mask showing button
        self.button_show_mask.clicked.connect(self.show_mask)

        # defining the shadow detection button
        self.button_detect_shadows.clicked.connect(self.detect_shadows)

        # defining the remove shadow button
        self.button_remove_shadows.clicked.connect(self.remove_shadows)

        # defining metric labels
        self.label_shadow_mask_mse = self.findChild(QLabel, "label_shadow_mask_mse")
        self.label_shadow_mask_mse.hide()

        self.label_image_mse = self.findChild(QLabel, "label_image_mse")
        self.label_image_mse.hide()

        self.label_image_psnr = self.findChild(QLabel, "label_image_psnr")
        self.label_image_psnr.hide()

        self.show()

    def choose_image(self):

        self.image_path = None
        self.gt_image = None
        self.gt_mask = None

        self.label_shadow_mask_mse.hide()
        self.label_image_mse.hide()
        self.label_image_psnr.hide()

        # opens file browser
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "./images", "Image files (*.jpg , *.png)")
        #self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "../SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages", "Image files (*.jpg , *.png)")
        #self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "../ISTD_Dataset/train/train_A", "Image files (*.jpg , *.png)")

        # outputs the path to the label
        if self.image_path:

            # displays the file's name
            image_file_name = self.image_path.split("/")[-1]

            self.image_file_name = image_file_name.split(".")[0]
            self.label_file_name.setText(image_file_name)

        # checking if there is a ground truth shadow mask
        if os.path.exists("shadow_masks/" + self.image_file_name + ".png") or os.path.exists("shadow_masks/" + self.image_file_name + ".jpg"):

            if os.path.exists("shadow_masks/" + self.image_file_name + ".png"):
                gt_mask_file = "shadow_masks/" + self.image_file_name + ".png"
            else:
                gt_mask_file = "shadow_masks/" + self.image_file_name + ".jpg"

            self.gt_mask = cv2.imread(gt_mask_file, cv2.IMREAD_GRAYSCALE)

        if os.path.exists("ground_truth_images/" + self.image_file_name + ".png") or os.path.exists("ground_truth_images/" + self.image_file_name + ".jpg"):

            if os.path.exists("ground_truth_images/" + self.image_file_name + ".png"):
                gt_image_file = "ground_truth_images/" + self.image_file_name + ".png"
            else:
                gt_image_file = "ground_truth_images/" + self.image_file_name + ".jpg"

            self.gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)

    def choose_mask(self):

        self.mask_path = None

        self.label_shadow_mask_mse.hide()
        self.label_image_mse.hide()
        self.label_image_psnr.hide()

        self.mask_path, _ = QFileDialog.getOpenFileName(self, "Choose Shadow Mask", "./shadow_masks", "Image files (*.jpg , *.png)")
        #self.mask_path, _ = QFileDialog.getOpenFileName(self, "Choose Shadow Mask", "../ISTD_Dataset/train/train_B", "Image files (*.jpg , *.png)")

        if self.image_path is None:
            print("You need to select an image first!")

        if self.mask_path and self.image_path:

            self.shadow_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

            # displays the file's name
            mask_file_name = self.mask_path.split("/")[-1]
            self.label_mask_name.setText(mask_file_name)

    def show_image(self):

        if self.image_path:

            cv2.imshow("Chosen Image", cv2.imread(self.image_path))

            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("You need to select an image!")
            return

    def show_mask(self):

        if self.shadow_mask is not None:

            cv2.imshow("Shadow mask", self.shadow_mask)

            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("You need to select a shadow mask or use one of the detection methods!")
            return

    def detect_shadows(self):

        if self.image_path is None:
            print("You need to select an image!")
            return
        if not self.radio_button_first_detection.isChecked() and not self.radio_button_second_detection.isChecked():
            print("You need to select a method!")
            return

        e1 = cv2.getTickCount()

        self.partial_results = self.check_box_partial_results.isChecked()

        if self.image_path and self.radio_button_first_detection.isChecked():
            self.shadow_mask = first_detection(self.image_path, self.partial_results)
            self.label_mask_name.setText("First shadow detection mask")

        if self.image_path and self.radio_button_second_detection.isChecked():
            self.shadow_mask = second_detection(self.image_path, self.partial_results)
            self.label_mask_name.setText("Second shadow detection mask")

        e2 = cv2.getTickCount()
        time1 = round((e2 - e1) / cv2.getTickFrequency(), 4)

        print("Shadow detection completed in: " + str(time1) + " seconds")
        print("-" * 50)

        # showing the shadow mask metric if there is a ground truth shadow mask in folder
        if self.gt_mask is not None:

            mask_mse = mean_square_error(self.shadow_mask, self.gt_mask)
            mask_rmse = round(sqrt(mask_mse), 2)

            self.label_shadow_mask_mse.setText("Shadow mask RMSE: " + str(mask_rmse))
            self.label_shadow_mask_mse.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def remove_shadows(self):

        if self.image_path is None:
            print("You need to select an image!")
            return

        if self.shadow_mask is None:
            print("You need to use one of the shadow detections!")
            return

        if not self.radio_button_first_removal.isChecked() and not self.radio_button_second_removal.isChecked():
            print("You need to select a method!")
            return

        if cv2.imread(self.image_path, cv2.IMREAD_COLOR).shape[0] != self.shadow_mask.shape[0]:
            print("The image and shadow mask is not the same size!")
            return

        e1 = cv2.getTickCount()

        self.partial_results = self.check_box_partial_results.isChecked()

        if self.radioButton_no_pp.isChecked():
            self.post_processing_operation = None

        if self.radioButton_median_blur_pp.isChecked():
            self.post_processing_operation = "median"

        if self.radioButton_inpainting_pp.isChecked():
            self.post_processing_operation = "inpainting"

        if self.image_path and self.radio_button_first_removal.isChecked():
            shadow_free = first_removal(self.image_path, self.shadow_mask, self.post_processing_operation, self.partial_results)

        if self.image_path and self.radio_button_second_removal.isChecked():
            shadow_free = second_removal(self.image_path, self.shadow_mask, self.post_processing_operation, self.partial_results)

        e2 = cv2.getTickCount()
        time1 = round((e2 - e1) / cv2.getTickFrequency(), 2)

        print("Shadow removal completed in: " + str(time1) + " seconds")
        print("-" * 50)

        # mean square error calculation for images
        if self.gt_mask is not None and self.gt_image is not None:

            shadow_free_shadow_only = shadow_free.copy()
            gt_image_shadow_only = self.gt_image.copy()

            shadow_free_shadow_only[self.gt_mask != 255] = 0
            gt_image_shadow_only[self.gt_mask != 255] = 0

            image_mse = mean_square_error(shadow_free_shadow_only, gt_image_shadow_only)
            image_rmse = round(sqrt(image_mse), 2)

            self.label_image_mse.setText("Image RMSE: " + str(image_rmse))
            self.label_image_mse.show()

            psnr = peak_signal_to_noise_ratio(image_mse)

            self.label_image_psnr.setText("Image PSNR: " + str(psnr) + "db")
            self.label_image_psnr.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication([])
    window = UI()
    app.exec()
