import cv2
import rclpy
import numpy as np
import time
import torch.nn as nn
import torch
import mediapipe as mp
import sys

import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *

from . import data_manager

import configparser
import os

from haejo_pkg.yolov5 import detect
from PIL import Image
from datetime import datetime as dt


config = configparser.ConfigParser()
config.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')
dev = config['dev']

from_class = uic.loadUiType(dev['GUI'])[0]

mp_pose = mp.solutions.pose
mp_pose_pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                            enable_segmentation = False, min_detection_confidence=0.7)
xy_list_list = []

attention_dot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
draw_line = [[0, 4], [0, 1], [4, 5], [1, 2], [5, 6], [2, 3], [6, 8], [3, 7], [9, 10]]


class MyDataset(Dataset):
    def __init__(self, seq_list):
        self.X = []
        self.y = []
        for dic in seq_list:
            self.y.append(dic['key'])
            self.X.append(dic['value'])
        
    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))
    
    def __len__(self):
        return len(self.X)


class skeleton_LSTM(nn.Module):
    def __init__(self):
        super(skeleton_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size= 26, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x


class DetectPhone(Node):
    def __init__(self):
        super().__init__('phone_detect')

        self.bridge = CvBridge()

        self.yolo = YOLO(dev['phone_yolo_model'])

        self.labels = self.yolo.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.labels))] 

        self.model = skeleton_LSTM()
        self.model.load_state_dict(torch.load(dev['phone_lstm_model'], map_location="cpu"))
        self.model.eval()
        print("success model load") 

    
    def pose_estimation(self, img):
        
        global xy_list_list

        length = 20
        dataset = []
        status = 'None'
        
        img = cv2.resize(img, (640, 640))

        results = mp_pose_pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks: 
            pass    

        else:
            xy_list = []
            idx = 0
            draw_line_dic = {}
            
            for x_and_y in results.pose_landmarks.landmark:
                if idx in attention_dot:
                    xy_list.append(x_and_y.x)
                    xy_list.append(x_and_y.y)
                    x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
                    draw_line_dic[idx] = [x, y]
                idx += 1
            xy_list_list.append(xy_list)

            for line in draw_line:
                x1, y1 = draw_line_dic[line[0]][0], draw_line_dic[line[0]][1]
                x2, y2 = draw_line_dic[line[1]][0], draw_line_dic[line[1]][1]
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

            
            phone_results = self.yolo(img, stream=True)

            for r in phone_results:
                
                annotator = Annotator(img)

                boxes = r.boxes
                
                for idx, box in enumerate(boxes):
                    b = box.xyxy[0]
                    c = box.cls
                
                    if int(c) == 67:
                        
                        color = self.colors[int(c)]
                        annotator.box_label(b, self.yolo.names[int(c)], color)

                        b_list = b.tolist()
                        
                        for idx in range(4):
                            xy_list.append(b_list[idx]/640.0)
                        break
                        
                    else:
                        if (len(boxes) - 1 == idx):
                            for _ in range(4):
                                xy_list.append(0.0)
                        else:
                            continue
                            
            img = annotator.result()

            if len(xy_list_list) == length:
                
                dataset = []
                dataset.append({'key' : 0, 'value' : xy_list_list})
                dataset = MyDataset(dataset)
                dataset = DataLoader(dataset)
                xy_list_list = []

                for data, label in dataset:
                    data = data.to("cuda")
                    
                    with torch.no_grad():
                        self.model = self.model.to("cuda")
                        result = self.model(data)
                        _, out = torch.max(result, 1)
                        
                        print(out.item())
                        if out.item() == 0: 
                            status = 'phone'
                            
                        else: 
                            status = 'work'

        return img, status
    

class DetectDesk(Node):
    def __init__(self):
        super().__init__('desk_detect')
        self.bridge = CvBridge()
        
    def detect_desk(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = Image.fromarray(img)
        img.save("./temp.jpg")
        img = detect.run(weights=dev['desk_yolo_model'], source="./temp.jpg")
        
        return img

class WindowClass(QMainWindow, from_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.bridge = CvBridge()

        self.isDetectPhoneOn = False
        self.detectphone = DetectPhone()
        
        self.isDetectDeskOn = False
        self.detectdesk = DetectDesk()
        
        self.detectphone.create_subscription(
        CompressedImage,
        '/image_raw/compressed',
        self.image_callback,
        1)
        
        self.detectdesk.create_subscription(
        CompressedImage,
        '/image_raw/compressed',
        self.image_callback,
        1)

        self.pixmap = QPixmap()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.spin_node)
        self.timer.start(100)

        '-----------camera-------------'
        self.detect_phone.clicked.connect(self.click_detect_phone)
        self.detect_desk.clicked.connect(self.click_detect_desk)


    def image_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        if self.isDetectPhoneOn == True:
            img, status = self.detectphone.pose_estimation(cv_image)
            cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            
        elif self.isDetectDeskOn == True:
            img = self.detectdesk.detect_desk(cv_image)
            self.writer.write(img)
        
        h,w,c = img.shape
        qimage = QImage(img.data, w, h, w*3, QImage.Format_BGR888)

        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())

        self.label.setPixmap(self.pixmap)


    def click_detect_phone(self):
        if self.isDetectPhoneOn == False:
            self.detect_phone.setText('stop')
            self.isDetectPhoneOn = True
            self.detect_light.hide()
            self.detect_door.hide()
            self.detect_desk.hide()
            self.detect_snack.hide()
            
            self.start_rec_and_req('detect_phone')


        else:
            self.detect_phone.setText('detect_phone')
            self.isDetectPhoneOn = False
            self.detect_light.show()
            self.detect_door.show()
            self.detect_desk.show()
            self.detect_snack.show()

            self.stop_rec_and_res()
        
        
    def click_detect_desk(self):
        if self.isDetectDeskOn == False:
            self.detect_desk.setText('stop')
            self.isDetectDeskOn = True
            self.detect_light.hide()
            self.detect_door.hide()
            self.detect_snack.hide()
            self.detect_phone.hide()
            
            self.start_rec_and_req('detect_desk')

        else:
            self.detect_desk.setText('detect_desk')
            self.isDetectDeskOn = False
            self.detect_light.show()
            self.detect_door.show()
            self.detect_snack.show()
            self.detect_phone.show()
            
            self.stop_rec_and_res()
            
            
    def start_rec_and_req(self, module):
        self.req_id = data_manager.insert_req(module)
            
        now = dt.now().strftime("%Y%m%d_%H%M")
        self.video_path = dev['video_dir'] + now + ".avi"
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.video_path, self.fourcc, 20.0, (640, 640))
        
        
    def stop_rec_and_res(self):
        self.writer.release()
        data_manager.insert_res(self.req_id, 'result_test', self.video_path)
            

    def spin_node(self):
        if self.isDetectPhoneOn == True:
            rclpy.spin_once(self.detectphone)
        elif self.isDetectDeskOn == True:
            rclpy.spin_once(self.detectdesk)
        # else:
            # print("not detect mode")
    

    def shutdown_ros(self):
        print("shutting down ROS")
        self.detectphone.destroy_node()
        rclpy.shutdown()  


def main(args=None):
    rclpy.init(args=None)

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.aboutToQuit.connect(myWindow.shutdown_ros)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()