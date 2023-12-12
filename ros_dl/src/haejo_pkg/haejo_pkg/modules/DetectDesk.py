from cv_bridge import CvBridge
from rclpy.node import Node
import cv2
from haejo_pkg.yolov5 import detect
from PIL import Image
import configparser
from haejo_pkg.utils import Logger


log = Logger.Logger('haejo_deep_learning.log')

config = configparser.ConfigParser()
config.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')
dev = config['dev']


class DetectDesk(Node):
    def __init__(self):
        super().__init__('desk_detect')
        self.bridge = CvBridge()
        
    def detect_desk(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = Image.fromarray(img)
        img.save("./temp.jpg")
        
        try:
            img = detect.run(weights=dev['desk_yolo_model'], source="./temp.jpg")
        except Exception as e:
            log.error(f" deep_learning detect_desk : {e}")
        
        return img