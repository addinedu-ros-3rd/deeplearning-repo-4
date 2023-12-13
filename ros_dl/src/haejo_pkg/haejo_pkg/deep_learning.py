import cv2
import rclpy
import configparser
import sys

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *

from . import data_manager
from datetime import datetime as dt
from haejo_pkg.utils import Logger
from haejo_pkg.utils.ConfigUtil import get_config

from haejo_pkg.modules.detect_door import Detectdoor
from haejo_pkg.modules.detect_light import Detectlight
from haejo_pkg.modules.detect_phone import DetectPhone
from haejo_pkg.modules.detect_snack import Detectsnack
from haejo_pkg.modules.detect_desk import DetectDesk

log = Logger.Logger('haejo_deep_learning.log')
config = get_config()

from_class = uic.loadUiType(config['GUI'])[0]


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
        self.detectdoor = Detectdoor()
        self.detectlight = Detectlight()
        self.detectsnack = Detectsnack()
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

        self.light_sub = self.detectlight.create_subscription(
        Image,
        '/image_raw',
        self.image_callback,
        1)
        self.light_sub

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
        
        '-------------DB---------------'
        self.set_combo()
        self.db_button_search.clicked.connect(self.search)
        self.db_tableWidget.itemDoubleClicked.connect(self.selectVideo)
        
        
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
        

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        if self.isDetectPhoneOn == True:
            img, status = self.detectphone.pose_estimation(cv_image)
            cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            
        elif self.isDetectDeskOn == True:
            img, result = self.detectdesk.detect_desk(cv_image)
            self.desk_result += result
            self.writer.write(img)

        elif self.isDetectDoorOn == True:
            img = self.detectdoor.detect_door(cv_image)

        elif self.isDetectSnackOn == True:
            img = self.detectsnack.detect_snack(cv_image)

        elif self.isDetectLightOn == True:
            img = self.detectlight.detect_light(cv_image)   
        
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

        else:
            self.fx_button_desk.setText('DESK')
            self.isDetectDeskOn = False
            self.fx_button_light.show()
            self.fx_button_door.show()
            self.fx_button_snack.show()
            self.fx_button_phone.show()
            
    
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
            

    def start_rec_and_req(self, module):
        self.req_id = data_manager.insert_req(module)
            
        now = dt.now().strftime("%Y%m%d_%H%M")
        self.video_path = config['video_dir'] + now + ".avi"
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.video_path, self.fourcc, 60.0, (640, 640))
        
        self.desk_result = ""
        
        
    def stop_rec_and_res(self, result):
        self.pixmap = QPixmap()
        self.video.setPixmap(self.pixmap)
        
        try:
            self.writer.release()
            data_manager.insert_res(self.req_id, result, self.video_path)
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
            rclpy.spin_once(self.detectlight)

        elif self.isDetectDoorOn == True:
            rclpy.spin_once(self.detectdoor)
        else:
            print("not detect mode")
    

    def shutdown_ros(self):
        print("shutting down ROS")

        self.detectphone.destroy_node()
        self.detectlight.destroy_node()
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