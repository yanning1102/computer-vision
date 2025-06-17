import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QComboBox, QGridLayout, QGroupBox, QLineEdit, QScrollArea, QMessageBox, QSlider, QDialog, QSpinBox
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
        self.alphabet_db = None  # Store alphabet database
        self.alphabet_db_vertical = None  # Store vertical alphabet database

    def initUI(self):
        # Set window title and size
        self.setWindowTitle('Camera Calibration and Augmented Reality')
        self.setGeometry(100, 100, 1200, 800)

        # Create GroupBox for each section
        load_image_group = QGroupBox("Load Image")
        calibration_group = QGroupBox("1. Calibration")
        augmented_reality_group = QGroupBox("2. Augmented Reality")
        stereo_disparity_group = QGroupBox("3. Stereo disparity map")
        sift_group = QGroupBox("4. SIFT")
        
        # Create buttons for Load Image section
        load_folder_button = QPushButton('Load Folder', self)
        load_image_l_button = QPushButton('Load Image_L', self)
        load_image_r_button = QPushButton('Load Image_R', self)
        
        load_image_layout = QVBoxLayout()
        load_image_layout.addWidget(load_folder_button)
        load_image_layout.addWidget(load_image_l_button)
        load_image_layout.addWidget(load_image_r_button)
        load_image_group.setLayout(load_image_layout)

        # Create buttons for Calibration section
        find_corners_button = QPushButton('1.1 Find Corners', self)
        find_intrinsic_button = QPushButton('1.2 Find Intrinsic', self)
        find_extrinsic_button = QPushButton('1.3 Find Extrinsic', self)
        self.extrinsic_image_index_spinbox = QSpinBox(self)
        self.extrinsic_image_index_spinbox.setRange(1, 15)
        find_distortion_button = QPushButton('1.4 Find Distortion', self)
        show_result_button = QPushButton('1.5 Show Result', self)

        calibration_layout = QGridLayout()
        calibration_layout.addWidget(find_corners_button, 0, 0)
        calibration_layout.addWidget(find_intrinsic_button, 1, 0)
        calibration_layout.addWidget(self.extrinsic_image_index_spinbox, 2, 0)
        calibration_layout.addWidget(find_extrinsic_button, 2, 1)
        calibration_layout.addWidget(find_distortion_button, 3, 0)
        calibration_layout.addWidget(show_result_button, 4, 0)
        calibration_group.setLayout(calibration_layout)

        # Create components for Augmented Reality section
        self.word_input = QLineEdit(self)
        show_words_board_button = QPushButton('2.1 Show words on board', self)
        show_words_vertical_button = QPushButton('2.2 Show words vertical', self)

        augmented_reality_layout = QVBoxLayout()
        augmented_reality_layout.addWidget(self.word_input)
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
        load_image_l_button.clicked.connect(self.load_image)
        load_image_r_button.clicked.connect(self.load_image)
        find_corners_button.clicked.connect(self.find_corners)
        find_intrinsic_button.clicked.connect(self.find_intrinsic)
        find_extrinsic_button.clicked.connect(self.find_extrinsic)
        find_distortion_button.clicked.connect(self.find_distortion)
        show_result_button.clicked.connect(self.show_result)
        show_words_board_button.clicked.connect(self.show_words_on_board)
        show_words_vertical_button.clicked.connect(self.show_words_vertical)
        stereo_disparity_button.clicked.connect(self.stereo_disparity_map)
        load_image1_button.clicked.connect(self.load_image)
        load_image2_button.clicked.connect(self.load_image)
        keypoints_button.clicked.connect(self.find_keypoints)
        matched_keypoints_button.clicked.connect(self.find_matched_keypoints)

    def load_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_name:
            self.image_folder = folder_name
            print(f'Loaded folder: {folder_name}')
            self.load_images_from_folder()

    def load_images_from_folder(self):
        self.corners_images = []
        for filename in sorted(os.listdir(self.image_folder)):
            if filename.endswith(('.bmp', '.jpg', '.png')):
                img_path = os.path.join(self.image_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    self.corners_images.append(img)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options)
        if file_name:
            print(f'Loaded image: {file_name}')

    def find_corners(self):
        print('Finding corners...')
        if not self.corners_images:
            print('No images loaded to find corners.')
            return

        chessboard_size = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.object_points = []
        self.image_points = []

        for img in self.corners_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.object_points.append(objp)
                self.image_points.append(corners)
                print('Corners found.')
            else:
                print('Corners not found in one of the images.')

    def find_intrinsic(self):
        print('Finding intrinsic parameters...')
        if not self.object_points or not self.image_points:
            print('No corners found to calibrate.')
            return

        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, (self.corners_images[0].shape[1], self.corners_images[0].shape[0]), None, None)
        if ret:
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.intrinsic_matrix = ins
            self.dist_coeffs = dist
            print(f'Intrinsic Matrix:{ins}')
            print(f'Distortion Coefficients:{dist}')
            # Calculate reprojection error for accuracy
            total_error = 0
            for i in range(len(self.object_points)):
                img_points2, _ = cv2.projectPoints(self.object_points[i], self.rvecs[i], self.tvecs[i], self.intrinsic_matrix, self.dist_coeffs)
                error = cv2.norm(self.image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            print(f'Total calibration error: {total_error / len(self.object_points)}')
        else:
            print('Calibration failed.')

    def show_words_on_board(self):
        print('Showing words on board...')
        word = self.word_input.text().upper()
        if len(word) > 6:
            print('Please enter a word with less than 6 characters.')
            return

        if self.intrinsic_matrix is None or self.dist_coeffs is None:
            print('Please perform camera calibration first.')
            return

        if not self.corners_images:
            print('No images loaded to project words.')
            return

        if self.alphabet_db is None:
            self.load_alphabet_database()

        # Define the positions for each letter based on the provided example
        positions = [
            [7, 5, 0],
            [4, 5, 0],
            [1, 5, 0],
            [7, 2, 0],
            [4, 2, 0],
            [1, 2, 0]
        ]

        for idx, (img, rvec, tvec) in enumerate(zip(self.corners_images, self.rvecs, self.tvecs)):
            img_copy = img.copy()
            for i, char in enumerate(word):
                if char in self.alphabet_db and i < len(positions):
                    # Adjust the 3D coordinates of each character
                    char_points_3d = np.array(self.alphabet_db[char], dtype=np.float32).reshape(-1, 3)
                    char_points_3d += np.array(positions[i])
                    char_points_3d = char_points_3d.reshape(-1, 1, 3)
                    img_points, _ = cv2.projectPoints(char_points_3d, rvec, tvec, self.intrinsic_matrix, self.dist_coeffs)
                    img_points = np.int32(img_points).reshape(-1, 2)
                    for j in range(0, len(img_points), 2):
                        if j + 1 < len(img_points):
                            cv2.line(img_copy, tuple(img_points[j]), tuple(img_points[j + 1]), (0, 0, 255), 6)  # Increase thickness to 4
            height, width, _ = img_copy.shape
            resized_image = cv2.resize(img_copy, (800, 600))  # Resize image to fit the window
            q_img = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], resized_image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            dialog = QDialog(self)
            dialog.setWindowTitle(f'Augmented Reality - Projected Word ({idx + 1})')
            label = QLabel(dialog)
            label.setPixmap(pixmap)
            dialog_layout = QVBoxLayout()
            dialog_layout.addWidget(label)
            dialog.setLayout(dialog_layout)
            dialog.exec_()

    def load_alphabet_database(self):
        print('Loading alphabet database...')
        self.alphabet_db = {}
        fs = cv2.FileStorage('alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            node = fs.getNode(letter)
            if not node.empty():
               self.alphabet_db[letter] = node.mat()
               print(f'Loaded letter {letter}: {self.alphabet_db[letter]}')  # 添加打印信息验证加载
            else:
              print(f'Letter {letter} not found in the database.')  # 如果未找到相应字母
        fs.release()

    def load_alphabet_vertical_database(self):
        print('Loading vertical alphabet database...')
        self.alphabet_db_vertical = {}
        fs = cv2.FileStorage('alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            node = fs.getNode(letter)
            if not node.empty():
                self.alphabet_db_vertical[letter] = node.mat()
                print(f'Loaded letter {letter} for vertical display: {self.alphabet_db_vertical[letter]}')  # 添加打印信息验证加载
            else:
                print(f'Letter {letter} not found in the vertical database.')  # 如果未找到相应字母
        fs.release()

    def find_extrinsic(self):
        print('Finding extrinsic parameters...')

    def find_distortion(self):
        print('Finding distortion parameters...')

    def show_result(self):
        print('Showing undistorted result...')

    def show_words_vertical(self):
        print('Showing words vertically...')
        word = self.word_input.text().upper()
        if len(word) > 6:
            print('Please enter a word with less than 6 characters.')
            return

        if self.intrinsic_matrix is None or self.dist_coeffs is None:
            print('Please perform camera calibration first.')
            return

        if not self.corners_images:
            print('No images loaded to project words.')
            return

        if self.alphabet_db_vertical is None:
            self.load_alphabet_vertical_database()

        # Define the positions for each letter based on the vertical layout
        positions = [
            [7, 5, 0],
            [4, 5, 0],
            [1, 5, 0],
            [7, 2, 0],
            [4, 2, 0],
            [1, 2, 0]
        ]

        for idx, (img, rvec, tvec) in enumerate(zip(self.corners_images, self.rvecs, self.tvecs)):
            img_copy = img.copy()
            for i, char in enumerate(word):
                if char in self.alphabet_db_vertical and i < len(positions):
                    # Adjust the 3D coordinates of each character
                    char_points_3d = np.array(self.alphabet_db_vertical[char], dtype=np.float32).reshape(-1, 3)
                    char_points_3d += np.array(positions[i])
                    char_points_3d = char_points_3d.reshape(-1, 1, 3)
                    img_points, _ = cv2.projectPoints(char_points_3d, rvec, tvec, self.intrinsic_matrix, self.dist_coeffs)
                    img_points = np.int32(img_points).reshape(-1, 2)
                    for j in range(0, len(img_points), 2):
                        if j + 1 < len(img_points):
                            cv2.line(img_copy, tuple(img_points[j]), tuple(img_points[j + 1]), (0, 255, 0), 6)  # Increase thickness to 4
            height, width, _ = img_copy.shape
            resized_image = cv2.resize(img_copy, (800, 600))  # Resize image to fit the window
            q_img = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], resized_image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            dialog = QDialog(self)
            dialog.setWindowTitle(f'Augmented Reality - Projected Word Vertically ({idx + 1})')
            label = QLabel(dialog)
            label.setPixmap(pixmap)
            dialog_layout = QVBoxLayout()
            dialog_layout.addWidget(label)
            dialog.setLayout(dialog_layout)
            dialog.exec_()

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
