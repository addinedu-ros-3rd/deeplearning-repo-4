import cv2
import rclpy
import sys
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *

from . import data_manager
from . import file_manager

from datetime import datetime as dt
from haejo_pkg.utils import Logger
from haejo_pkg.utils.ConfigUtil import get_config

from haejo_pkg.modules.detect_door import DetectDoor
# from haejo_pkg.modules.detect_light import DetectLight
from haejo_pkg.modules.detect_phone import DetectPhone
from haejo_pkg.modules.detect_snack import DetectSnack
from haejo_pkg.modules.detect_desk import DetectDesk


log = Logger.Logger('haejo_deep_learning.log')
config = get_config()

from_class = uic.loadUiType(config['GUI'])[0]


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

        '---------flag------------'
        self.isDetectPhoneOn = False
        self.isDetectSnackOn = False
        self.isDetectLightOn = False
        self.isDetectDoorOn = False
        self.isDetectDeskOn = False        

        self.detectphone = DetectPhone()
        self.detectdoor = DetectDoor()
        # self.detectlight = DetectLight()
        self.detectsnack = DetectSnack()
        self.detectdesk = DetectDesk()
        
        '--------subscription----------'
        self.phone_sub = self.detectphone.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)
        self.phone_sub

        self.door_sub = self.detectdoor.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)
        self.door_sub

        # self.light_sub = self.detectlight.create_subscription(
        # Image,
        # '/image_raw',
        # self.image_callback,
        # 1)
        # self.light_sub

        self.snack_sub = self.detectsnack.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)
        self.snack_sub

        self.desk_sub = self.detectdesk.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)
        self.desk_sub

        '--------------utils---------'
        self.pixmap = QPixmap()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.spin_node)
        self.timer.start(10)

        '-----------camera-------------'
        self.fx_button_phone.clicked.connect(self.click_detect_phone)
        self.fx_button_desk.clicked.connect(self.click_detect_desk)
        self.fx_button_snack.clicked.connect(self.click_detect_snack)
        self.fx_button_door.clicked.connect(self.click_detect_door)
        self.fx_button_light.clicked.connect(self.click_detect_light)
        
        self.record = Camera(self)
        self.record.deamon = True
        self.record.update.connect(self.updateRecording)
        
        '-------------DB---------------'
        self.set_combo()
        self.db_button_search.clicked.connect(self.search)
        self.db_tableWidget.itemDoubleClicked.connect(self.selectVideo)
        self.fx_button_play.clicked.connect(self.controlVideo)
        
        '-------------UI---------------'
        self.white_button.clicked.connect(self.change_to_white)
        self.dark_button.clicked.connect(self.change_to_black)
        self.db_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.zeroto255 = [self.fx_button_play, self.groupBox, self.fx_button_phone, self.fx_button_door,self.fx_button_light, 
                          self.fx_button_snack, self.fx_button_desk, self.groupBox_2, self.db_comboBox, self.groupBox_3, 
                          self.db_date_from, self.db_date_to, self.db_label_for,self.db_button_search,self.db_tableWidget, self.title_label]

    
    def change_colors(self, color_rgb):
        for target in self.zeroto255:
            target.setStyleSheet(f"color: {color_rgb};")
    
    def change_to_white(self):
        self.change_colors("rgb(0, 0, 0)")
        self.setStyleSheet("background-color: rgb(245, 245, 245);")
        self.label_2.setStyleSheet("background-color: rgb(222, 221, 218);")
        self.video.setStyleSheet("background-color: rgb(255, 255, 255); ")
        self.label.setStyleSheet("background-color: rgb(222, 221, 218);  ")
        
    def change_to_black(self):
        self.change_colors("rgb(255, 255, 255)") 
        self.setStyleSheet("background-color: rgb(34, 33, 39);")
        self.label_2.setStyleSheet("background-color: rgb(50, 45, 58);")
        self.video.setStyleSheet("background-color: rgb(0, 0, 0); ")
        self.label.setStyleSheet("background-color: rgb(50, 45, 58);  ")
        
        
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
        self.videoCapture = cv2.VideoCapture(file)
        
        if self.videoCapture.isOpened():  # VideoCapture 인스턴스 생성 확인
            self.showThumbnail()
            
            
    def showThumbnail(self):
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
            self.detect_result += status
            self.updateRecording(img)
            
        elif self.isDetectDeskOn == True:
            img, result = self.detectdesk.detect_desk(cv_image)
            self.detect_result += result
            self.updateRecording(img)

        elif self.isDetectDoorOn == True:
            img = self.detectdoor.detect_door(cv_image)
            self.updateRecording(img)

        elif self.isDetectSnackOn == True:
            img = self.detectsnack.detect_snack(cv_image)
            self.updateRecording(img)

        elif self.isDetectLightOn == True:
            # img = self.detectlight.detect_light(cv_image)
            # self.updateRecording(img)
            log.info("detect light 주석 하고 테스트")
        
        log.info(img.shape)
        
        h, w, c = img.shape
        qimage = QImage(img.data, w, h, w*3, QImage.Format_BGR888)

        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.label.width(), self.label.height())

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

            self.stop_rec_and_res(self.phone_status)
        
        
    def click_detect_desk(self):
        if self.isDetectDeskOn == False:
            self.fx_button_desk.setText('STOP')
            self.isDetectDeskOn = True
            self.fx_button_light.hide()
            self.fx_button_door.hide()
            self.fx_button_snack.hide()
            self.fx_button_phone.hide()
            
            self.start_rec_and_req('DESK')

        else:
            self.fx_button_desk.setText('DESK')
            self.isDetectDeskOn = False
            self.fx_button_light.show()
            self.fx_button_door.show()
            self.fx_button_snack.show()
            self.fx_button_phone.show()
            
            self.stop_rec_and_res(self.detect_result)
            
    
    def click_detect_door(self):
        if self.isDetectDoorOn == False:
            self.fx_button_door.setText('STOP')
            self.isDetectDoorOn = True
            self.fx_button_light.hide()
            self.fx_button_phone.hide()
            self.fx_button_desk.hide()
            self.fx_button_snack.hide()

            self.start_rec_and_req('DOOR')
        
        else:
            self.fx_button_door.setText('DOOR')
            self.isDetectDoorOn = False
            self.fx_button_light.show()
            self.fx_button_phone.show()
            self.fx_button_desk.show()
            self.fx_button_snack.show()
            
            self.stop_rec_and_res('DOOR RESULT')  # to do: 실제 인식 결과 기록 필요


    def click_detect_light(self):
        if self.isDetectLightOn == False:
            self.fx_button_light.setText('STOP')
            self.isDetectLightOn = True
            self.fx_button_phone.hide()
            self.fx_button_door.hide()
            self.fx_button_desk.hide()
            self.fx_button_snack.hide()

            self.start_rec_and_req('LIGHT')

        else:
            self.fx_button_light.setText('LIGHT')
            self.isDetectLightOn = False
            self.fx_button_phone.show()
            self.fx_button_door.show()
            self.fx_button_desk.show()
            self.fx_button_snack.show()
            
            self.stop_rec_and_res('LIGHT RESULT')  # to do: 실제 인식 결과 기록 필요


    def click_detect_snack(self):
        if self.isDetectSnackOn == False:
            self.fx_button_snack.setText('STOP')
            self.isDetectSnackOn = True
            self.fx_button_light.hide()
            self.fx_button_door.hide()
            self.fx_button_desk.hide()
            self.fx_button_phone.hide()

            self.start_rec_and_req('SNACK')

        else:
            self.fx_button_snack.setText('SNACK')
            self.isDetectSnackOn = False
            self.fx_button_light.show()
            self.fx_button_door.show()
            self.fx_button_desk.show()
            self.fx_button_phone.show()
            
            self.stop_rec_and_res('SNACK RESULT')  # to do: 실제 인식 결과 기록 필요
            

    def start_rec_and_req(self, module):
        self.req_id = data_manager.insert_req(module)
            
        now = dt.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = config['video_dir'] + now + ".avi"
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.video_path, self.fourcc, 20.0, (640, 640))
        
        self.detect_result = ""
        
        
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

        elif self.isDetectSnackOn == True:
            rclpy.spin_once(self.detectsnack)

        elif self.isDetectLightOn == True:
            # rclpy.spin_once(self.detectlight)
            log.info("detect light 주석 하고 테스트")

        elif self.isDetectDoorOn == True:
            rclpy.spin_once(self.detectdoor)
        else:
            log.info("not detect mode")
    

    def shutdown_ros(self):
        log.info("shutting down ROS")

        self.detectphone.destroy_node()
        # self.detectlight.destroy_node()
        self.detectdoor.destroy_node()
        self.detectsnack.destroy_node()
        self.detectdesk.destroy_node()

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