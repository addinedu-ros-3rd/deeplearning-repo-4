from rclpy.node import Node
from ultralytics.utils.plotting import Annotator
from haejo_pkg.utils import Logger
from haejo_pkg.utils.ConfigUtil import get_config
import cv2
import torch
import numpy as np


log = Logger.Logger('detect_desk.py')
config = get_config()


class DetectDesk(Node):
    def __init__(self):
        super().__init__('desk_detect')
        
        self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=config['desk_model'], force_reload=True)

        # 모델을 추론 모드로 설정
        self.model.eval()
        
    def detect_desk(self, img):
        
        desk_result = ""

        # 추론 수행
        results = self.model(img)
        log.info(results)
        
        labels = self.model.names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))] 
        
        annotator = Annotator(img)
        
        for i, det in enumerate(results.xyxy[0]):
            score = float(det[4])  # 감지된 객체의 확률
            log.info(score)
            if score >= 0.5:  # 확률이 0.5 이상인 경우에만 박스 그리기
                label = int(det[5])  # 객체의 클래스 인덱스
                box = det[:4].cpu().numpy()  # 박스 좌표

                # 박스 그리기
                color = colors[label]
                annotator.box_label(box, self.model.names[label], color)
                desk_result = self.model.names[label]
        
        img = annotator.result()
        
        img = cv2.resize(img, (640, 640))  # display 크기 통일
        
        return img, desk_result