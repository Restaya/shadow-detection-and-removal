from PyQt5.QtWidgets import *
from PyQt5 import uic

import cv2
import methods.first_method


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Loading the UI file
        uic.loadUi("menu.ui", self)

        self.button_choose_image = self.findChild(QPushButton, "push_button_choose_image")
        self.button_show_image = self.findChild(QPushButton, "push_button_show_image")

        self.label_file_name = self.findChild(QLabel, "label_file_name")

        self.image_path = None

        # defining the choose image button
        self.button_choose_image.clicked.connect(self.choose_image)

        # defining the image showing button
        self.button_show_image.clicked.connect(self.show_image)

        self.show()

    def choose_image(self):

        # opens file browser
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image files (*.jpg , *.png)")

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

            

if __name__ == "__main__":
    app = QApplication([])
    window = UI()
    app.exec()
