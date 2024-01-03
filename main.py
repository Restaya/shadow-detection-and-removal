from PyQt5.QtWidgets import *
from PyQt5 import uic

import cv2
from methods import first_method,second_method


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # loading the UI file
        uic.loadUi("menu.ui", self)

        # defining the buttons
        self.button_choose_image = self.findChild(QPushButton, "push_button_choose_image")
        self.button_show_image = self.findChild(QPushButton, "push_button_show_image")

        self.button_detect_shadows = self.findChild(QPushButton, "push_button_detect_shadows")
        self.button_remove_shadows = self.findChild(QPushButton, "push_button_remove_shadows")

        self.label_file_name = self.findChild(QLabel, "label_file_name")

        # defining radio buttons
        self.radio_button_first_detection = self.findChild(QRadioButton, "radio_button_first_detection")
        self.radio_button_second_detection = self.findChild(QRadioButton, "radio_button_second_detection")

        self.radio_button_first_removal = self.findChild(QRadioButton, "radio_button_first_removal")
        self.radio_button_second_removal = self.findChild(QRadioButton, "radio_button_second_removal")

        self.image_path = None
        self.shadow_mask = None

        # defining the choose image button
        self.button_choose_image.clicked.connect(self.choose_image)

        # defining the image showing button
        self.button_show_image.clicked.connect(self.show_image)

        # defining the shadow detection button
        self.button_detect_shadows.clicked.connect(self.detect_shadows)

        # defining the remove shadow button
        self.button_remove_shadows.clicked.connect(self.remove_shadows)

        self.show()

    def choose_image(self):

        # opens file browser
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "./test_pictures", "Image files (*.jpg , *.png)")

        # outputs the path to the label
        if self.image_path:

            # displays the file's name
            file_name = str.split(self.image_path, "/")[-1]
            self.label_file_name.setText(file_name)

    def show_image(self):

        if self.image_path:

            cv2.imshow("Chosen Image", cv2.imread(self.image_path))

            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("You need to select an image!")

    def detect_shadows(self):

        if self.image_path and self.radio_button_first_detection.isChecked():
            self.shadow_mask = first_method.first_method_detection(self.image_path)

        if self.image_path and self.radio_button_second_detection.isChecked():
            self.shadow_mask = second_method.second_method_detection(self.image_path)

        if self.image_path is None:
            print("You need to select an image!")
        if self.radio_button_first_detection.isChecked() == False and self.radio_button_second_detection.isChecked() == False:
            print("You need to select a method!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def remove_shadows(self):

        if self.image_path and self.radio_button_first_removal.isChecked():
            first_method.first_method_removal(self.image_path, self.shadow_mask)

        if self.image_path and self.radio_button_second_removal.isChecked():
            print("Work in progress!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if self.image_path is None:
            print("You need to select an image!")
        if self.shadow_mask is None:
            print("You need to use one of the shadow detections!")
        if self.radio_button_first_removal.isChecked() == False and self.radio_button_second_removal.isChecked() == False:
            print("You need to select a method!")


if __name__ == "__main__":
    app = QApplication([])
    window = UI()
    app.exec()
