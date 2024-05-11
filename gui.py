import cv2
from PyQt5.QtWidgets import *
from PyQt5 import uic

from shadow_detection import *
from shadow_removal import *


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
        self.mask_path = None
        self.shadow_mask = None
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

        self.show()

    def choose_image(self):

        # opens file browser
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "./images", "Image files (*.jpg , *.png)")
        #self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "../SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages", "Image files (*.jpg , *.png)")

        # outputs the path to the label
        if self.image_path:

            # displays the file's name
            image_file_name = str.split(self.image_path, "/")[-1]
            self.label_file_name.setText(image_file_name)

            #self.shadow_mask = None

    def choose_mask(self):
        self.mask_path, _ = QFileDialog.getOpenFileName(self, "Choose Shadow Mask", "./shadow_masks", "Image files (*.jpg , *.png)")

        if self.image_path is None:
            print("You need to select an image first!")

        if self.mask_path and self.image_path:

            self.shadow_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

            # displays the file's name
            mask_file_name = str.split(self.mask_path, "/")[-1]
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

        if self.image_path and self.radio_button_second_detection.isChecked():
            self.shadow_mask = second_detection(self.image_path, self.partial_results)

        e2 = cv2.getTickCount()
        time1 = round((e2 - e1) / cv2.getTickFrequency(), 4)

        print("Shadow detection completed in: " + str(time1) + " seconds")
        print("------------------------------------------------------------------")

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

        e1 = cv2.getTickCount()

        self.partial_results = self.check_box_partial_results.isChecked()

        if self.radioButton_no_pp.isChecked():
            self.post_processing_operation = None

        if self.radioButton_median_blur_pp.isChecked():
            self.post_processing_operation = "median"

        if self.radioButton_inpainting_pp.isChecked():
            self.post_processing_operation = "inpainting"

        if self.image_path and self.radio_button_first_removal.isChecked():
            first_removal(self.image_path, self.shadow_mask, self.post_processing_operation, self.partial_results)

        if self.image_path and self.radio_button_second_removal.isChecked():
            second_removal(self.image_path, self.shadow_mask, self.post_processing_operation, self.partial_results)

        e2 = cv2.getTickCount()
        time1 = round((e2 - e1) / cv2.getTickFrequency(), 4)

        print("Shadow removal completed in: " + str(time1) + " seconds")
        print("------------------------------------------------------------------")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication([])
    window = UI()
    app.exec()
