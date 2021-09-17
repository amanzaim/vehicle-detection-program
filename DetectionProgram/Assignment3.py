import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
import cv2
import numpy as np
import os
import tensorflow as tf
import csv

from packaging import version
from collections import defaultdict
from io import StringIO
from PIL import Image

# object detection import
from utils import label_map_util
from utils import visualization_utils as vis_util

# initialize .csv file to be used as records for information log
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])


# initialize class to be used for information log
class vehicleInformation:
    def __init__(self, carnum, movement, speed, color, cartype):
        self.carnum = carnum
        self.movement = movement
        self.speed = speed
        self.color = color
        self.cartype = cartype


# initialize list to be used for vehicleinformation
vehiclelist = []

# initialize counter
total_passed_vehicle = 0

# initialize model to be used
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# creation of path to frozen detection graph. This is the model used to detect vehicles
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of strings to add correct labels for each object detected.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading a frozen tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
                                              3)).astype(np.uint8)


# object detection function
def detect_vehicles(video):
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:

            # define the tensors for detection graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # define the detection boxes where each detected objects are found
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # gives a score, class and
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # initialize a new integer to be used as carnum
            i = 0

            # opens the video and runs the detection program
            while video.isOpened():
                (ret, frame) = video.read()

                if not ret:
                    print('end of the video file...')
                    break

                input_frame = frame

                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # detection of vehicles
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                              detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        video.get(1),
                        input_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=3,
                    )

                total_passed_vehicle += counter

                # insert vehicle counter text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Vehicles Detected: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    font,
                )

                # modify the cv2.line everytime a vehicle is detected passing through it
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                    vehiclelist.append(vehicleInformation('car ' + str(i), direction,
                                                          str(speed).split(".")[0], color, size))
                    i += 1
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                # insert the text above the cv2.line
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA
                )

                # insert the information text about last vehicle detected in a rectangle in the video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )
                cv2.putText(
                    input_frame,
                    '-Movement Direction: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Speed(km/h): ' + str(speed).split(".")[0],
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Color: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )
                cv2.putText(
                    input_frame,
                    '-Vehicle Size/Type: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )

                cv2.imshow('Vehicle Detection Program', input_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, color, direction, speed) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            video.release()
            cv2.destroyAllWindows()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("GUI/GUI.ui", self)

        self.startButton.clicked.connect(self.Start)
        self.browseButton.clicked.connect(self.browse)

    # Button function to browse files
    def browse(self):
        data_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', r"D:\Programming", '*.*')
        self.videoInput.setText(data_path)

    # Button function to output video to QGraphicsView and start the Vehicle Detection Program
    def Start(self):
        path = self.videoInput.text()
        print(path)

        video = cv2.VideoCapture(path)
        detect_vehicles(video)

        del vehiclelist[0]

        # to put the vehiclelist class into the information log
        for obj in vehiclelist:
            carnum = obj.carnum
            move = obj.movement
            speed = obj.speed
            color = obj.color
            cartype = obj.cartype

            self.informationLog.append(carnum + ", Movement : " + move + ", Speed : " +
                                       speed + ", Color : " + color + ", Car Type : " + cartype)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
