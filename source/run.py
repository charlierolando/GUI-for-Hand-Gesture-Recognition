from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QComboBox, QLineEdit, QTextEdit
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import argparse
import cv2
import os
import serial
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

camera = 2  # 0 laptop camera
scale_percent = 100  # percent of original size

global stop_flag
global connect_flag
global x_old
x_old = "@"
global opencr


class HandGesture(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        # self._run_flag = False

    def run(self):
        global stop_flag
        stop_flag = True
        global connect_flag
        connect_flag = False

        while True:
            if app.exec_() == False:
                break

            while stop_flag == True:
                continue

            camera = a.combo_box.currentText()

            default_model_dir = ''
            default_model = 'HandGesture_edgetpu.tflite'
            default_labels = 'labels.txt'
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', help='.tflite model path',
                                default=os.path.join(default_model_dir, default_model))
            parser.add_argument('--labels', help='label file path',
                                default=os.path.join(default_model_dir, default_labels))
            parser.add_argument('--top_k', type=int, default=3,
                                help='number of categories with highest score to display')
            parser.add_argument('--camera_idx', type=int,
                                help='Index of which video source to use. ', default=camera)
            parser.add_argument('--threshold', type=float, default=0.8,
                                help='classifier score threshold')
            args = parser.parse_args()

            print('Loading {} with {} labels.'.format(args.model, args.labels))
            interpreter = make_interpreter(args.model)
            interpreter.allocate_tensors()
            labels = read_label_file(args.labels)
            inference_size = input_size(interpreter)

            cap = cv2.VideoCapture(args.camera_idx)

            # while cap.isOpened() & self._run_flag:
            while cap.isOpened():

                if stop_flag == True:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                cv2_im = frame

                cv2_im = cv2.flip(cv2_im, 1) # flip # mirror
                
                cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

                run_inference(interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(interpreter, args.threshold)[:args.top_k]
                cv2_im = append_objs_to_img(
                    a, cv2_im, inference_size, objs, labels)

                # Edited #
                width = int(cv2_im.shape[1] * scale_percent / 100)
                height = int(cv2_im.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                cv2_im = cv2.resize(cv2_im, dim, interpolation=cv2.INTER_AREA)
                # Edited #

                # while self._run_flag:
                # ret, cv2_img = cv2_im
                # if ret:
                self.change_pixmap_signal.emit(cv2_im)

                # cv2.imshow('frame', cv2_im)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            # cap = cv2.VideoCapture(0)
            # shut down capture system
            cap.release()
            cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()

    def play(self):
        """Sets run flag to Truse and waits for thread to finish"""
        global stop_flag
        stop_flag = False
        a.black.setStyleSheet("background-color: none;")
        a.button1.setText("Stop")
        print("Play")
        # self.wait()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        global stop_flag
        stop_flag = True
        a.black.setStyleSheet("background-color: black;")
        a.button1.setText("Start")
        print("Stop")
        # self.wait()

    def disconn(self):
        global connect_flag
        connect_flag = False

        global opencr
        if (opencr.isOpen()):
            opencr.close()

        a.button3.setText("Connect")
        print("Disconnect")
        # self.wait()

    def conn(self):
        global connect_flag
        connect_flag = True

        global opencr
        opencr = serial.Serial(a.combo_box2.currentText(),
                               1000000, timeout=.1)  # connect to openCR

        a.button3.setText("Disconnect")
        print("Connecting...")
        print("Connect")
        # self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()

        # self._run_flag = False

        self.setWindowTitle("Hand Gesture")
        # self.resize(885, 510)
        self.setFixedSize(885, 510)
        self.display_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)

        # self.loading_label = QLabel("Loading...", self)
        # self.loading_label.setStyleSheet("background-color: white; color: black;")
        # self.loading_label.move(12, 12)
        # self.loading_label.resize(70, 20)

        self.black = QLabel(self)
        self.black.setStyleSheet("background-color: black;")
        self.black.move(11, 15)
        self.black.resize(self.display_width, self.display_height)

        # create a text label
        # self.textLabel = QLabel('Hand Gesture Detection')
        # self.textLabel.move(self.display_width + 11, self.display_height + 11)

        pos_x = 600 + 64
        pos_y = 32
        combo_box_button_dis = 40

        # combo box
        self.combo_box_label = QLabel("Camera port:", self)
        self.combo_box_label.setStyleSheet("color: black;")
        self.combo_box_label.move(pos_x, pos_y - 25)

        self.combo_box = QComboBox(self)
        self.combo_box.move(pos_x, pos_y)
        self.combo_box.resize(210, 30)
        my_list = ["0", "1", "2", "3"]
        self.combo_box.addItems(my_list)

        self.combo_box2_label = QLabel("openCR port:", self)
        self.combo_box2_label.setStyleSheet("color: black;")
        self.combo_box2_label.move(pos_x, pos_y + 70)

        self.combo_box2 = QComboBox(self)
        self.combo_box2.move(pos_x, pos_y + 95)
        self.combo_box2.resize(210, 30)
        my_list2 = ["/dev/ttyACM0", "/dev/ttyACM1"]
        self.combo_box2.addItems(my_list2)

        # Text Browser & Text Edit
        self.lineedit_label = QLabel("Result:", self)
        self.lineedit_label.setStyleSheet("color: black;")
        self.lineedit_label.move(pos_x, pos_y + 165)
        self.lineedit = QLineEdit(self)
        self.lineedit.resize(210, 30)
        self.lineedit.move(pos_x, pos_y + 190)
        self.textedit = QTextEdit(self)
        self.textedit.resize(210, 190)
        self.textedit.move(pos_x, pos_y + 230)

        # button
        self.button1 = QPushButton(self)
        self.button1.setText("Start")
        self.button1.resize(210, 30)
        self.button1.move(pos_x, combo_box_button_dis + pos_y)

        self.button1.clicked.connect(self.button1_clicked)

        self.button2 = QPushButton(self)
        self.button2.setText("Clear")
        self.button2.resize(210, 30)
        self.button2.move(pos_x, combo_box_button_dis + pos_y + 390)

        self.button2.clicked.connect(self.button2_clicked)

        self.button3 = QPushButton(self)
        self.button3.setText("Connect")
        self.button3.resize(210, 30)
        self.button3.move(pos_x, combo_box_button_dis + pos_y + 95)

        self.button3.clicked.connect(self.button3_clicked)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = HandGesture()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def button1_clicked(self):
        global stop_flag
        if (stop_flag == True):
            HandGesture.play(self)
        else:
            HandGesture.stop(self)

    def button2_clicked(self):
        self.textedit.clear()
        self.lineedit.clear()

    def button3_clicked(self):
        # global opencr
        # if (opencr.isOpen()):
        
        global connect_flag
        if (connect_flag == True):
            HandGesture.disconn(self)
        else:
            HandGesture.conn(self)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def append_objs_to_img(self, cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        self.textedit.append("{}".format(label))
        self.lineedit.setText(label)
        # print(label)

        global opencr
        global connect_flag
        if (connect_flag == True):
            write_opencr(labels.get(obj.id, obj.id))
            # if (opencr.isOpen()):
            #     write_opencr(labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def write_opencr(x):
    global x_old
    if x == x_old:
        return
    x_old = x
    # print(x)

    global opencr
    opencr.write(bytes(x, 'utf-8'))
    print(x, opencr.readline())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
