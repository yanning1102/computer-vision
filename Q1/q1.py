import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QComboBox, QGridLayout, QGroupBox, QLineEdit, QScrollArea, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize, QTimer

class CalibrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_folder = None
        self.corners_images = []
        self.current_image_index = 0
        self.object_points = []  # 3D points in real world space
        self.image_points = []  # 2D points in image plane
        self.rvecs = []  # Rotation vectors
        self.tvecs = []  # Translation vectors
        self.intrinsic_matrix = None  # Store intrinsic matrix
        self.dist_coeffs = None  # Store distortion coefficients

    def initUI(self):
        # Set window title and size
        self.setWindowTitle('Camera Calibration and Augmented Reality')
        self.setGeometry(100, 100, 1000, 800)

        # Create GroupBox for each section
        load_image_group = QGroupBox("Load Image")
        calibration_group = QGroupBox("1. Calibration")
        augmented_reality_group = QGroupBox("2. Augmented Reality")
        stereo_disparity_group = QGroupBox("3. Stereo disparity map")
        sift_group = QGroupBox("4. SIFT")

        # Create buttons for Load Image section
        load_folder_button = QPushButton('Load Folder', self)
        load_image_L_button = QPushButton('Load Image_L', self)
        load_image_R_button = QPushButton('Load Image_R', self)

        load_image_layout = QVBoxLayout()
        load_image_layout.addWidget(load_folder_button)
        load_image_layout.addWidget(load_image_L_button)
        load_image_layout.addWidget(load_image_R_button)
        load_image_group.setLayout(load_image_layout)

        # Create buttons for Calibration section
        find_corners_button = QPushButton('1.1 Find Corners', self)
        find_intrinsic_button = QPushButton('1.2 Find Intrinsic', self)
        self.find_extrinsic_combo = QComboBox(self)
        self.find_extrinsic_combo.addItems([str(i) for i in range(1, 16)])
        find_extrinsic_button = QPushButton('1.3 Find Extrinsic', self)
        find_distortion_button = QPushButton('1.4 Find Distortion', self)
        show_result_button = QPushButton('1.5 Show Result', self)
        
        calibration_layout = QGridLayout()
        calibration_layout.addWidget(find_corners_button, 0, 0)
        calibration_layout.addWidget(find_intrinsic_button, 1, 0)
        calibration_layout.addWidget(QLabel("1.3 Find Extrinsic"), 2, 0)
        calibration_layout.addWidget(self.find_extrinsic_combo, 2, 1)
        calibration_layout.addWidget(find_extrinsic_button, 2, 2)
        calibration_layout.addWidget(find_distortion_button, 3, 0)
        calibration_layout.addWidget(show_result_button, 4, 0)
        calibration_group.setLayout(calibration_layout)

        # Create components for Augmented Reality section
        word_input = QLineEdit(self)
        show_words_board_button = QPushButton('2.1 Show words on board', self)
        show_words_vertical_button = QPushButton('2.2 Show words vertical', self)

        augmented_reality_layout = QVBoxLayout()
        augmented_reality_layout.addWidget(word_input)
        augmented_reality_layout.addWidget(show_words_board_button)
        augmented_reality_layout.addWidget(show_words_vertical_button)
        augmented_reality_group.setLayout(augmented_reality_layout)

        # Create buttons for Stereo Disparity Map section
        stereo_disparity_button = QPushButton('3.1 Stereo disparity map', self)

        stereo_disparity_layout = QVBoxLayout()
        stereo_disparity_layout.addWidget(stereo_disparity_button)
        stereo_disparity_group.setLayout(stereo_disparity_layout)

        # Create buttons for SIFT section
        load_image1_button = QPushButton('Load Image1', self)
        load_image2_button = QPushButton('Load Image2', self)
        keypoints_button = QPushButton('4.1 Keypoints', self)
        matched_keypoints_button = QPushButton('4.2 Matched Keypoints', self)

        sift_layout = QVBoxLayout()
        sift_layout.addWidget(load_image1_button)
        sift_layout.addWidget(load_image2_button)
        sift_layout.addWidget(keypoints_button)
        sift_layout.addWidget(matched_keypoints_button)
        sift_group.setLayout(sift_layout)

        # Create main layout and add all sections
        main_layout = QGridLayout()
        main_layout.addWidget(load_image_group, 0, 0)
        main_layout.addWidget(calibration_group, 0, 1)
        main_layout.addWidget(augmented_reality_group, 0, 2)
        main_layout.addWidget(stereo_disparity_group, 1, 0)
        main_layout.addWidget(sift_group, 1, 1)

        # Set layout for main window
        self.setLayout(main_layout)

        # Connect buttons to corresponding functions
        load_folder_button.clicked.connect(self.load_folder)
        load_image_L_button.clicked.connect(self.load_image)
        load_image_R_button.clicked.connect(self.load_image)
        find_corners_button.clicked.connect(self.find_corners)
        find_intrinsic_button.clicked.connect(self.find_intrinsic)
        find_extrinsic_button.clicked.connect(self.find_extrinsic)
        find_distortion_button.clicked.connect(self.find_distortion)
        show_result_button.clicked.connect(self.show_result_comparison)
        show_words_board_button.clicked.connect(self.show_words_on_board)
        show_words_vertical_button.clicked.connect(self.show_words_vertical)
        stereo_disparity_button.clicked.connect(self.stereo_disparity_map)
        load_image1_button.clicked.connect(self.load_image)
        load_image2_button.clicked.connect(self.load_image)
        keypoints_button.clicked.connect(self.find_keypoints)
        matched_keypoints_button.clicked.connect(self.find_matched_keypoints)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options)
        if file_name:
            print(f'Loaded image: {file_name}')

    def load_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_name:
            self.image_folder = folder_name
            print(f'Loaded folder: {folder_name}')

    def find_corners(self):
        if not self.image_folder:
            print('No folder loaded')
            return

        # Define chessboard size
        chessboard_size = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Prepare 3D chessboard points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        self.corners_images = []
        self.object_points = []
        self.image_points = []
        self.current_image_index = 0

        # Iterate through images in the folder
        for filename in sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))) if x.split('.')[0].isdigit() else x):
            if filename.endswith(('.bmp', '.jpg', '.png')):
                img_path = os.path.join(self.image_folder, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

                if ret:
                    # Refine corner positions
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                    self.corners_images.append(img)
                    self.object_points.append(objp)
                    self.image_points.append(corners)
                    print(f'Corners found in {filename}')
                else:
                    print(f'Corners not found in {filename}')

        # Display processed result
        self.show_corners()

    def show_corners(self):
        if self.corners_images:
            for i, img in enumerate(self.corners_images):
                height, width = img.shape[:2]
                bytes_per_line = 3 * width
                qformat = QImage.Format_RGB888
                qimg = QImage(img.data, width, height, bytes_per_line, qformat).rgbSwapped()
                pixmap = QPixmap.fromImage(qimg)
                message = QMessageBox(self)
                message.setIconPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
                message.setWindowTitle(f"Corners Found in Image {i + 1}")
                message.setStandardButtons(QMessageBox.NoButton)
                QTimer.singleShot(1000, message.accept)  # Auto close after 1 second
                message.exec_()

    
    def find_intrinsic(self):
        if not self.object_points or not self.image_points:
            print('No corners found to calibrate.')
            return

        # Calibrate camera using found 3D and 2D points
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, (self.corners_images[0].shape[1], self.corners_images[0].shape[0]), None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO)
        if ret:
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.intrinsic_matrix = ins
            self.dist_coeffs = dist
            np.set_printoptions(precision=6, suppress=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Intrinsic Matrix")
            msg.setText(f'Intrinsic Matrix (camera matrix):\n{ins}')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            print(f'Intrinsic Matrix:\n{ins}')
            print(f'Distortion Coefficients:\n{dist}')
        else:
            print('Calibration failed.')

    def find_extrinsic(self):
        if not self.rvecs or not self.tvecs:
            print('No extrinsic data available. Please run intrinsic calibration first.')
            return

        index = self.find_extrinsic_combo.currentIndex()
        if index >= len(self.rvecs):
            print('Invalid index selected for extrinsic calculation.')
            return

        # Compute rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(self.rvecs[index])
        tvec = self.tvecs[index]

        # Concatenate rotation matrix and translation vector to form extrinsic matrix
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        np.set_printoptions(precision=6, suppress=True)
        extrinsic_matrix_message = QMessageBox()
        extrinsic_matrix_message.setIcon(QMessageBox.Information)
        extrinsic_matrix_message.setWindowTitle("Extrinsic Matrix")
        extrinsic_matrix_message.setText(f'Extrinsic Matrix (Image {index + 1}):\n{extrinsic_matrix}')
        extrinsic_matrix_message.setStandardButtons(QMessageBox.Ok)
        extrinsic_matrix_message.exec_()

    def find_distortion(self):
        if self.dist_coeffs is not None:
            np.set_printoptions(precision=6, suppress=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Distortion Coefficients")
            msg.setText(f'Distortion Coefficients:\n{self.dist_coeffs}')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            print(f'Distortion Coefficients:\n{self.dist_coeffs}')
        else:
            print('No distortion coefficients available. Please run intrinsic calibration first.')

    def show_result_comparison(self):
        if self.intrinsic_matrix is None or self.dist_coeffs is None:
            print('No intrinsic matrix or distortion coefficients available. Please run intrinsic calibration first.')
            return

        index = self.find_extrinsic_combo.currentIndex()
        if index >= len(self.corners_images):
            print('Invalid index selected for result comparison.')
            return

        img = self.corners_images[index]
        undistorted_img = cv2.undistort(img, self.intrinsic_matrix, self.dist_coeffs, None, self.intrinsic_matrix)

            # Concatenate original and undistorted images side by side for comparison
        comparison_img = np.hstack((img, undistorted_img))
        height, width = comparison_img.shape[:2]
        bytes_per_line = 3 * width
        qformat = QImage.Format_RGB888
        qimg = QImage(comparison_img.data, width, height, bytes_per_line, qformat).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        message = QMessageBox(self)
        message.setIconPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        message.setWindowTitle(f"Undistorted Result")
        message.setStandardButtons(QMessageBox.Ok)
        message.exec_()

    def show_words_on_board(self):
        print('Showing words on board...')

    def show_words_vertical(self):
        print('Showing words vertically...')

    def stereo_disparity_map(self):
        print('Generating stereo disparity map...')

    def find_keypoints(self):
        print('Finding keypoints...')

    def find_matched_keypoints(self):
        print('Finding matched keypoints...')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CalibrationApp()
    ex.show()
    sys.exit(app.exec_())
