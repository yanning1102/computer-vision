import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QMessageBox, QGridLayout, QGroupBox, QComboBox, QLineEdit, QDialog)
from PyQt5.QtGui import QPixmap

class SIFTKeypointsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.image_path_2 = None

    def initUI(self):
        main_layout = QGridLayout()

        # Load Image Group
        load_group = QGroupBox("Load Image")
        load_layout = QVBoxLayout()
        self.loadFolderButton = QPushButton('Load folder', self)
        load_layout.addWidget(self.loadFolderButton)
        self.loadImageLButton = QPushButton('Load Image_L', self)
        self.loadImageLButton.clicked.connect(self.loadImage)
        load_layout.addWidget(self.loadImageLButton)
        self.loadImageRButton = QPushButton('Load Image_R', self)
        self.loadImageRButton.clicked.connect(self.loadImage2)
        load_layout.addWidget(self.loadImageRButton)
        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group, 0, 0)

        # Calibration Group
        calibration_group = QGroupBox("1.Calibration")
        calibration_layout = QVBoxLayout()
        self.findCornersButton = QPushButton('1.1 Find corners', self)
        calibration_layout.addWidget(self.findCornersButton)
        self.findIntrinsicButton = QPushButton('1.2 Find intrinsic', self)
        calibration_layout.addWidget(self.findIntrinsicButton)
        self.extrinsicComboBox = QComboBox(self)
        self.extrinsicComboBox.addItems([str(i) for i in range(1, 16)])
        calibration_layout.addWidget(self.extrinsicComboBox)
        self.findExtrinsicButton = QPushButton('1.3 Find extrinsic', self)
        calibration_layout.addWidget(self.findExtrinsicButton)
        self.findDistortionButton = QPushButton('1.4 Find distortion', self)
        calibration_layout.addWidget(self.findDistortionButton)
        self.showResultButton = QPushButton('1.5 Show result', self)
        calibration_layout.addWidget(self.showResultButton)
        calibration_group.setLayout(calibration_layout)
        main_layout.addWidget(calibration_group, 0, 1)

        # Augmented Reality Group
        ar_group = QGroupBox("2.Augmented Reality")
        ar_layout = QVBoxLayout()
        self.arInputBox = QLineEdit(self)
        ar_layout.addWidget(self.arInputBox)
        self.showVortexOnBoardButton = QPushButton('2.1 Show vortex on board', self)
        ar_layout.addWidget(self.showVortexOnBoardButton)
        self.showVortexVerticalButton = QPushButton('2.2 Show vortex vertical', self)
        ar_layout.addWidget(self.showVortexVerticalButton)
        ar_group.setLayout(ar_layout)
        main_layout.addWidget(ar_group, 0, 2)

        # Stereo Disparity Map Group
        stereo_group = QGroupBox("3.Stereo disparity map")
        stereo_layout = QVBoxLayout()
        self.stereoDisparityMapButton = QPushButton('3.1 Stereo disparity map', self)
        stereo_layout.addWidget(self.stereoDisparityMapButton)
        stereo_group.setLayout(stereo_layout)
        main_layout.addWidget(stereo_group, 1, 0)

        # SIFT Group
        sift_group = QGroupBox("4.SIFT")
        sift_layout = QVBoxLayout()
        self.loadImageButton = QPushButton('Load Image 1', self)
        self.loadImageButton.clicked.connect(self.loadImage)
        sift_layout.addWidget(self.loadImageButton)

        self.loadImage2Button = QPushButton('Load Image 2', self)
        self.loadImage2Button.clicked.connect(self.loadImage2)
        sift_layout.addWidget(self.loadImage2Button)

        self.keypointsButton = QPushButton('4.1 Keypoints', self)
        self.keypointsButton.clicked.connect(self.showKeypoints)
        sift_layout.addWidget(self.keypointsButton)

        self.matchedKeypointsButton = QPushButton('4.2 Matched Keypoints', self)
        self.matchedKeypointsButton.clicked.connect(self.showMatchedKeypoints)
        sift_layout.addWidget(self.matchedKeypointsButton)
        sift_group.setLayout(sift_layout)
        main_layout.addWidget(sift_group, 1, 1)

        # Set the layout for the main window
        self.setLayout(main_layout)
        self.setWindowTitle('SIFT Keypoints and Calibration UI')
        self.setGeometry(100, 100, 1000, 800)

    def loadImage(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name

    def loadImage2(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path_2 = file_name

    def showKeypoints(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        # Load the image in grayscale
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            QMessageBox.critical(self, "Error", "Failed to load the image!")
            return

        # Convert image to grayscale (if not already)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect SIFT keypoints
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Draw keypoints on the image with specific color (green)
        output_image = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))

        # Resize the output image to fit in the dialog
        height, width = output_image.shape[:2]
        max_height, max_width = 300, 400
        scale = min(max_width / width, max_height / height)
        resized_image = cv2.resize(output_image, (int(width * scale), int(height * scale)))

        # Save the result
        result_path = 'sift_keypoints_result.jpg'
        cv2.imwrite(result_path, resized_image)

        # Display the result in a popup window
        dialog = QDialog(self)
        dialog.setWindowTitle("SIFT Keypoints Result")
        dialog.setGeometry(100, 100, 400, 300)
        dialog_layout = QVBoxLayout()
        result_label = QLabel()
        pixmap = QPixmap(result_path)
        result_label.setPixmap(pixmap)
        dialog_layout.addWidget(result_label)
        dialog.setLayout(dialog_layout)
        dialog.exec_()

    def showMatchedKeypoints(self):
        if not self.image_path or not self.image_path_2:
            QMessageBox.warning(self, "Warning", "Please load both images first!")
            return

        # Load the images in grayscale
        image1 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.image_path_2, cv2.IMREAD_GRAYSCALE)
        if image1 is None or image2 is None:
            QMessageBox.critical(self, "Error", "Failed to load one or both images!")
            return

        # Detect SIFT keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Draw the matched keypoints between the two images
        matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize the matched image to fit in the dialog
        height, width = matched_image.shape[:2]
        max_height, max_width = 300, 400
        scale = min(max_width / width, max_height / height)
        resized_image = cv2.resize(matched_image, (int(width * scale), int(height * scale)))

        # Save the result
        result_path = 'sift_matched_keypoints_result.jpg'
        cv2.imwrite(result_path, resized_image)

        # Display the result in a popup window
        dialog = QDialog(self)
        dialog.setWindowTitle("SIFT Matched Keypoints Result")
        dialog.setGeometry(100, 100, 400, 300)
        dialog_layout = QVBoxLayout()
        result_label = QLabel()
        pixmap = QPixmap(result_path)
        result_label.setPixmap(pixmap)
        dialog_layout.addWidget(result_label)
        dialog.setLayout(dialog_layout)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SIFTKeypointsApp()
    ex.show()
    sys.exit(app.exec_())
