import sys
import cv2
import time
import datetime
import torch.nn as nn
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *

mp_pose = mp.solutions.pose
attention_dot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
draw_line = [[0, 4], [0, 1], [4, 5], [1, 2], [5, 6], [2, 3], [6, 8], [3, 7], [9, 10]]
from_class = uic.loadUiType("haejo.ui")[0]

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
        self.lstm1 = nn.LSTM(input_size= 22, hidden_size=128, num_layers=1, batch_first=True)
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


class WindowClass(QMainWindow, from_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.isCameraOn = False
        self.isRecStart = False
        '--------hide btn--------'
        self.recordbtn.hide()  
        self.capturebtn.hide()
        self.video_stopbtn.hide()

        self.pixmap = QPixmap()

        self.cam_thread = Camera(self)
        self.cam_thread.daemon = True

        self.record = Camera(self)
        self.record.daemon = True

        self.vid = Camera(self)
        self.vid.daemon = True

        self.open_File.clicked.connect(self.openFile)
        

        '-----------camera-------------'
        self.camerabtn.clicked.connect(self.clickCamera)
        self.cam_thread.update.connect(self.updateCamera)

        '-----------record-------------'
        self.recordbtn.clicked.connect(self.clickRecord)
        self.record.update.connect(self.updateRecord)

        '-----------capture------------'
        self.capturebtn.clicked.connect(self.capture)

        '-----------video---------'
        self.vid.update.connect(self.updateVideo)
        self.video_stopbtn.clicked.connect(self.clickVideo)
    
    
    def openFile(self):
        file = QFileDialog.getOpenFileName(self, 'open file', './')

        if file[0].split('.')[1] in ['avi', 'mp4']: 
            self.video = cv2.VideoCapture(file[0])
            self.vid.running = True
            self.vid.start()
            self.video_stopbtn.show()

        else:
            image = cv2.imread(file[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h,w,c = image.shape
            qimage = QImage(image.data, w, h, w*c, QImage.Format_RGB888)

            self.pixmap = self.pixmap.fromImage(qimage)
            self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())

            self.label.setPixmap(self.pixmap)
    
    def clickRecord(self):
        if self.isRecStart == False:
            self.recordbtn.setText('Rec Stop')
            self.isRecStart = True

            self.recordingStart()

        else:                          
            self.recordbtn.setText('Rec Start')
            self.isRecStart = False

            self.recordingStop()
    
    def recordingStart(self):
        self.record.running = True
        self.record.start()

        '--------record start----------'
        self.now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.now + '.avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.writer = cv2.VideoWriter(filename, self.fourcc, 20.0, (w,h))
    
    def recordingStop(self):
        self.record.running = False

        '----------record stop----------'
        if self.isRecStart == True:
            self.writer.release()
    
    def clickCamera(self):
        if self.isCameraOn == False:
            self.camerabtn.setText('Camera Off')
            self.isCameraOn = True
            self.recordbtn.show()
            self.capturebtn.show()

            self.cameraStart()
        else:
            self.camerabtn.setText('Camera On')
            self.isCameraOn = False
            self.recordbtn.hide()
            self.capturebtn.hide()

            self.cameraStop()
            self.recordingStop()  # if camera off, record video


    def cameraStart(self):
        self.cam_thread.running = True
        self.cam_thread.start() # start Thread
        self.camera = cv2.VideoCapture(0)
        self.load_model()


    def load_model(self):
        ## load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load("../model/aa.pt", map_location=self.device)
        self.model.eval()


    def cameraStop(self):
        self.cam_thread.running = False
        self.camera.release
        cv2.destroyAllWindows()
    
    def updateCamera(self):
        length = 20

        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            
        ret, img = self.camera.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            dataset = []
            status = 'None'
            # pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
            #                     enable_segmentation = False, min_detection_confidence=0.3)
            img = cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            h,w,c = img.shape

            qimage = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
            
            self.pixmap = self.pixmap.fromImage(qimage)
            self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())

            self.label.setPixmap(self.pixmap)
                
        
    
    def updateRecord(self):
        retval, image = self.camera.read()
        if retval:
            self.writer.write(image)

    def updateVideo(self):
        
        retval, image = self.video.read()

        if retval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h,w,c = image.shape
            qimage = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
            
            self.pixmap = self.pixmap.fromImage(qimage)
            self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())

            self.label.setPixmap(self.pixmap)
            
    
    def clickVideo(self):
        self.vid.running = False
        self.video.release
        
    
    def capture(self):
        retval, image = self.camera.read()
        if retval:
            self.now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.now + '.png'
            cv2.imwrite(filename, image)


class Camera(QThread):  # 매 1초마다 시그널을 보내는 쓰레드를 만듬
    update = pyqtSignal()

    def __init__(self, sec=0, parent=None):
        super().__init__()
        self.main = parent
        self.running = True
    
    def run(self):
        count = 0
        while self.running == True:
            self.update.emit()  # make signal
            time.sleep(0.1)
    
    def stop(self):
        self.running = False




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())