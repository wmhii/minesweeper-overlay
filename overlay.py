import sys
import statistics as stats
import os

import pyscreenshot as ImageGrab
import numpy as np
import cv2

from PIL import Image
from PyQt5 import QtGui, QtCore, QtWidgets, QtTest

BOMB = 14
SAFE = 13
UNKNOWN = 15
CLEAR = 0

debugging = False


class MyMainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.NoDropShadowWindowHint
        )
        self.setWindowFlag(QtCore.Qt.CoverWindow, True)
        self.setWindowFlag(QtCore.Qt.WindowTransparentForInput, True)

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        # self.setAttribute(QtCore.Qt., True)

        self.templates = list()
        for temp in os.listdir('templates'):
            img = cv2.imread(f'templates/{temp}', cv2.IMREAD_GRAYSCALE)
            self.templates.append((os.path.splitext(temp)[0], img))

        self.last_frame = None
        self.setGeometry(*self.find_sweeper())
        grid_w, grid_h, self.cell_size = self.determine_grid_details()

        self.game_grid = np.ones((grid_w, grid_h), dtype=np.uint8) * 15
        self.read_grid()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(250)

    def update(self):
        super().update()
        # self.hide()
        self.find_sweeper()
        self.read_grid()
        # self.show()
        self.solve_grid_iteration()
        self.timer.start(100)

    def reset_grid(self):
        self.game_grid[:, :] = UNKNOWN
        self.read_grid()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(painter.Antialiasing)
        green = QtGui.QColor(0, 0, 255, 128)
        red = QtGui.QColor(255, 0, 0, 128)

        painter.setPen(QtGui.QColor(0, 0, 0, 200))
        painter.setBrush(QtGui.QColor(0, 255, 0, 128))

        cell_w, cell_h = self.game_grid.shape
        for x in range(cell_w):
            for y in range(cell_h):
                x_pos = x*self.cell_size
                y_pos = y*self.cell_size
                item = self.game_grid[x, y]
                if item == BOMB:
                    painter.fillRect(y_pos+2, x_pos+2, 10, 10, red)
                elif item == SAFE:
                    painter.fillRect(y_pos+2, x_pos+2, 10, 10, green)
                elif 0 < item < 9 or item == UNKNOWN or item == CLEAR:
                    pass
                else:
                    painter.drawText(y_pos + self.cell_size // 2, x_pos + self.cell_size // 2,
                                     str(self.game_grid[x, y]))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        QtWidgets.QApplication.exit(0)

    def find_sweeper(self):
        # Grab a screenshot and load it into a np array
        # grab = ImageGrab.grab().convert('RGB')
        grab = QtWidgets.QApplication.primaryScreen().grabWindow(0).toImage()
        # grab = Image.new('RGB', (1200, 1200))
        w = grab.size().width()
        h = grab.size().height()

        record = [('b', 'u1'), ('g', 'u1'), ('r', 'u1'), ('unused', 'u1')]
        b = grab.bits()
        b.setsize(w*h*4)
        img = np.frombuffer(b, np.uint8).reshape((h, w, 4))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Make it grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a threshold to help find where the box is
        (thresh, threshold_img) = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

        # Grab the Contour with the biggest area
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)

        # Return the coords
        x, y, w, h = cv2.boundingRect(biggest)
        self.last_frame = img[y:y+h, x:x+w]

        return x, y, w, h

    def determine_grid_details(self):
        """Figures out the cell_size in pixels and the width and height of the grid in cells"""
        img = self.last_frame

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 0)

        edges = cv2.Canny(blur, 5, 10)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        distances = []
        for i in range(len(contours) - 1):
            cnt = contours[i]
            cnt2 = contours[i + 1]

            x, _, _, _ = cv2.boundingRect(cnt)
            x2, _, _, _ = cv2.boundingRect(cnt2)

            distances.append(abs(x - x2))

        cell_size = stats.mode(distances)
        w, h, _ = img.shape
        width = w//cell_size
        height = h//cell_size

        return width, height, cell_size

    def determine_cell_state(self, grey_cell, hsv_cell):
        """Returns the state of the cell"""
        detailed = grey_cell

        highest = (0, None)
        for template in self.templates:
            if template[1].shape != grey_cell.shape:
                continue
            res = cv2.matchTemplate(detailed, template[1], cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > highest[0] and score > 0.30:
                highest = (score, template[0])

        if highest[1] is None:
            if hsv_cell[self.cell_size-1, self.cell_size-1] == 255:
                highest = (1, 'undiscovered')
            else:
                highest = (1, 'clear')

        return highest

    def read_grid(self):
        """Scan through the cells and estimate what they are"""
        grey = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        h_ret, hsv_thresh = cv2.threshold(hsv, 90, 255, cv2.THRESH_BINARY_INV)

        ret, thresh = cv2.threshold(grey, 160, 255, cv2.THRESH_BINARY_INV)

        size = self.cell_size
        w, h = grey.shape

        for x in range(0, h, size):
            for y in range(0, w, size):
                cell = thresh[y:y+size, x:x+size]
                hsv_cell = hsv_thresh[y:y+size, x:x+size]

                match_percent, state = self.determine_cell_state(cell, hsv_cell)

                if 'one' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 1
                elif 'two' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 2
                elif 'three' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 3
                elif 'four' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 4
                elif 'five' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 5
                elif 'six' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 6
                elif 'seven' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 7
                elif 'eight' in state:
                    self.game_grid[y//self.cell_size, x//self.cell_size] = 8
                elif 'flag' in state:
                    # Ignore flags, we should know if something is a bomb!
                    # self.game_grid[y//self.cell_size, x//self.cell_size] = UNKNOWN
                    pass
                elif state == 'undiscovered':
                    pass
                    # self.game_grid[y//self.cell_size, x//self.cell_size] = UNKNOWN
                elif state == 'clear':
                    self.game_grid[y//self.cell_size, x//self.cell_size] = CLEAR
                elif state is None:
                    pass
                    # self.game_grid[y//self.cell_size, x//self.cell_size] = 20

    def solve_grid_iteration(self):
        w, h = self.game_grid.shape

        # All the indeces of 1-8s
        indeces = np.argwhere((0 < self.game_grid) & (self.game_grid < 9))

        for x, y in indeces:
            low_x = max(0, x - 1)
            high_x = min(w, x + 2)

            low_y = max(0, y - 1)
            high_y = min(h, y + 2)

            kernel = self.game_grid[low_x:high_x, low_y:high_y]
            num = self.game_grid[x, y]
            # self.game_grid[x, y] = 20
            # QtWidgets.QApplication.processEvents()
            # self.repaint()

            # count how many undiscovered cells/bombs there are around the cell
            # count how many bombs there are, if the bomb count is the same as the number
            # then the undiscovered unknown cells are safe cells

            bomb_count = np.count_nonzero(kernel == BOMB)
            unknown_count = np.count_nonzero(kernel == UNKNOWN)
            safe_count = np.count_nonzero(kernel == SAFE)

            # Check for errors
            if bomb_count > num or bomb_count + unknown_count < num:
                kernel[kernel == BOMB] = UNKNOWN
                kernel[kernel == SAFE] = UNKNOWN

            if bomb_count == num:
                kernel[kernel == UNKNOWN] = SAFE
            else:
                # Otherwise if the number of unknowns+bombs = the number,
                # then all unknowns are bombs (this will consider safe cells)
                if bomb_count + unknown_count == num:
                    kernel[kernel == UNKNOWN] = BOMB

            # self.game_grid[x, y] = num


class EditWindow(QtWidgets.QWidget):

    def __init__(self, overlay):
        super().__init__()
        self.overlay: MyMainWindow = overlay

        self.setLayout(QtWidgets.QVBoxLayout())

        reset = QtWidgets.QPushButton("Reset")
        self.layout().addWidget(reset)

        flag_button = QtWidgets.QPushButton("Mark Flags")
        self.layout().addWidget(flag_button)
        flag_button.clicked.connect(self.mark_flags)

        reset.clicked.connect(self.reset_grid)

    def reset_grid(self):
        self.overlay.reset_grid()

    def mark_flags(self):
        click_here = self.overlay.pos() + QtCore.QPoint(10, 10)
        r = QtTest.QTest.mouseClick(self.overlay, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(10, 10))
        print(r)
        screen = QtWidgets.QApplication.primaryScreen()
        cursor = QtGui.QCursor()
        cursor.setPos(click_here)


app = QtWidgets.QApplication(sys.argv)
window = MyMainWindow()
settings_window = EditWindow(window)
window.show()
settings_window.show()
app.exec()
