import cv2
import rclpy
import numpy as np
import torch.nn as nn
import torch
import mediapipe as mp
import sys
import time

from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *

from . import data_manager
from . import file_manager

from datetime import datetime as dt
from haejo_pkg.modules import DetectDesk
from haejo_pkg.utils import Logger
from haejo_pkg.utils.ConfigUtil import get_config


config = get_config()
from_class = uic.loadUiType(config['GUI'])[0]

log = Logger.Logger('haejo_deep_learning.log')

mp_pose = mp.solutions.pose
mp_pose_pose = mp_pose.Pose(static_image_mode=False, 
                            model_complexity=1,
                            enable_segmentation=False, 
                            min_detection_confidence=0.7)

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

        self.yolo = YOLO(config['phone_yolo_model'])

        self.labels = self.yolo.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.labels))] 

        self.model = skeleton_LSTM()
        self.model.load_state_dict(torch.load(config['phone_lstm_model'], map_location="cpu"))
        self.model.eval()
        log.info("success model load")

    
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
    
    
class Camera(QThread):
    update = pyqtSignal()

    def __init__(self, sec=0, parent=None):
        super().__init__()
        self.main = parent
        self.running = True


    def run(self):
        count = 0
        while self.running == True:
            self.update.emit()
            time.sleep(0.05)

    def stop(self):
        self.running == False


class WindowClass(QMainWindow, from_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.bridge = CvBridge()
        
        self.record = Camera(self)
        self.record.deamon = True
        self.record.update.connect(self.updateRecording)
        
        self.play = Camera(self)
        self.play.daemon = True

        self.isDetectPhoneOn = False
        self.detectphone = DetectPhone()
        
        self.isDetectDeskOn = False
        self.detectdesk = DetectDesk.DetectDesk()
        
        self.detectphone.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)

        self.detectdesk.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)

        self.pixmap = QPixmap()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.spin_node)
        self.timer.start(10)

        '-----------camera-------------'
        self.fx_button_phone.clicked.connect(self.click_detect_phone)
        self.fx_button_desk.clicked.connect(self.click_detect_desk)
        
        '-------------DB---------------'
        self.set_combo()
        self.db_button_search.clicked.connect(self.search)
        self.db_tableWidget.itemDoubleClicked.connect(self.selectVideo)
        self.fx_button_play.clicked.connect(self.controlVideo)
        
        
    def set_combo(self):
        moduleList = data_manager.select_module()
        self.db_comboBox.addItem("ALL")
        for item in moduleList:
            self.db_comboBox.addItem(item[0])
            
            
    def search(self):
        self.db_tableWidget.setRowCount(0)
        
        module = self.db_comboBox.currentText()
        start_date = self.db_date_from.date().toString("yyyy-MM-dd")
        end_date = self.db_date_to.date().toString("yyyy-MM-dd")
        
        result = data_manager.select_video(module, start_date, end_date)
        
        for row in result:
            resultRow = self.db_tableWidget.rowCount()
            self.db_tableWidget.insertRow(resultRow)
            for i, v in enumerate(row):
                self.db_tableWidget.setItem(resultRow, i, QTableWidgetItem(str(v)))
                
        header = self.db_tableWidget.horizontalHeader()
        
        for col in range(self.db_tableWidget.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            
            
    def selectVideo(self, clickedItem):
        idx = self.db_tableWidget.model().index(clickedItem.row(), 4)
        file = self.db_tableWidget.model().data(idx)
        log.info(file)
        
        self.videoCapture = cv2.VideoCapture(file)
        
        if self.videoCapture.isOpened():  # VideoCapture 인스턴스 생성 확인
            self.showThumbnail()
            
            
    def showThumbnail(self):
        self.pixmap = QPixmap()
        self.video.setPixmap(self.pixmap)
        
        ret, frame = self.videoCapture.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w, c = frame.shape
            bytes_per_line = 3 * w
            self.qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.pixmap = QPixmap.fromImage(self.qimage)
            self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())
            
            self.video.setPixmap(self.pixmap)
            
    
    def controlVideo(self):
        if self.fx_button_play.text() == "▶":
            self.playVideo()
        else:
            self.pauseVideo()
            
            
    def playVideo(self):
        self.isVideoEnd = False
        self.fx_button_play.setText("❚❚")
        
        while self.isVideoEnd == False:
            ret, frame = self.videoCapture.read()
            
            if not ret:
                self.isVideoEnd = True
                self.fx_button_play.setText("▶")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w, c = frame.shape
            bytes_per_line = 3 * w
            self.qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.pixmap = QPixmap.fromImage(self.qimage)
            self.video.setPixmap(self.pixmap)
            
            QApplication.processEvents()  # prevent GUI freeze

            time.sleep(0.05)  # 저장된 동영상에 맞게 setting

        self.videoCapture.release()
        
        
    def pauseVideo(self):
        self.isVideoEnd = True
        self.fx_button_play.setText("▶")


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        if self.isDetectPhoneOn == True:
            img, status = self.detectphone.pose_estimation(cv_image)
            cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            self.updateRecording(img)
            
        elif self.isDetectDeskOn == True:
            img, result = self.detectdesk.detect_desk(cv_image)
            self.desk_result += result
            self.updateRecording(img)
        
        log.info(img.shape)
        
        h, w, c = img.shape
        qimage = QImage(img.data, w, h, w*3, QImage.Format_BGR888)

        self.pixmap = self.pixmap.fromImage(qimage)

        self.video.setPixmap(self.pixmap)


    def click_detect_phone(self):
        if self.isDetectPhoneOn == False:
            self.fx_button_phone.setText('STOP')
            self.isDetectPhoneOn = True
            self.fx_button_light.hide()
            self.fx_button_door.hide()
            self.fx_button_desk.hide()
            self.fx_button_snack.hide()
            
            self.start_rec_and_req('PHONE')


        else:
            self.fx_button_phone.setText('PHONE')
            self.isDetectPhoneOn = False
            self.fx_button_light.show()
            self.fx_button_door.show()
            self.fx_button_desk.show()
            self.fx_button_snack.show()

            self.stop_rec_and_res("WORK")  # to do: 인식 결과 받도록 수정
        
        
    def click_detect_desk(self):
        if self.isDetectDeskOn == False:
            self.fx_button_desk.setText('STOP')
            self.isDetectDeskOn = True
            self.fx_button_light.hide()
            self.fx_button_door.hide()
            self.fx_button_snack.hide()
            self.fx_button_phone.hide()
            
            self.start_rec_and_req('DESK')
            self.desk_result = ""

        else:
            self.fx_button_desk.setText('DESK')
            self.isDetectDeskOn = False
            self.fx_button_light.show()
            self.fx_button_door.show()
            self.fx_button_snack.show()
            self.fx_button_phone.show()
            
            self.stop_rec_and_res(self.desk_result)
            
            
    def start_rec_and_req(self, module):
        self.req_id = data_manager.insert_req(module)
            
        # now = dt.now().strftime("%Y%m%d_%H%M%S")
        # self.video_path = config['video_dir'] + now + ".avi"
        
        self.filename = dt.now().strftime("%Y%m%d_%H%M%S") + ".avi"
        self.local_path = config['video_dir'] + self.filename
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.local_path, self.fourcc, 20.0, (640, 640))
        
        self.desk_result = ""
        
        
    def updateRecording(self, img):
        self.writer.write(img)
        
        
    def stop_rec_and_res(self, result):
        self.pixmap = QPixmap()
        self.video.setPixmap(self.pixmap)
        
        try:
            self.writer.release()
            s3_uploaded = file_manager.s3_put_object(self.local_path, self.filename)
            
            if s3_uploaded:
                url = f"https://haejo.s3.ap-northeast-2.amazonaws.com/{self.local_path}"
                data_manager.insert_res(self.req_id, result, url)
        
        except Exception as e:
            log.error(f" deep_learning stop_rec_and_res : {e}")
            

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