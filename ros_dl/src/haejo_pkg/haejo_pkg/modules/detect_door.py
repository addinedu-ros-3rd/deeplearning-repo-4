import cv2
import torch
from rclpy.node import Node
from ultralytics.utils.plotting import Annotator
from haejo_pkg.utils.ConfigUtil import get_config
from haejo_pkg.utils import Logger

log = Logger.Logger('detect_door.py')
config = get_config()

class DetectDoor(Node):
    def __init__(self):
        super().__init__('door_detect')

        self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=config['door_model'], force_reload=False)

        # 모델을 추론 모드로 설정
        self.model.eval()
        
        log.info("success detect_door model load")        
    
    def detect_door(self, img):
        
        door_result = ""

        # 추론 수행
        results = self.model(img)
        log.info(results)

        if results.xyxy[0].tolist() == []:
            cv2.putText(img, "NOT DETECT DOOR", (320,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        # 각 객체에 대해 결과 확인
        annotator = Annotator(img)
        
        for i, det in enumerate(results.xyxy[0]):
            score = float(det[4])  # 감지된 객체의 확률
            log.info(score)
            if score >= 0.5:  # 확률이 0.5 이상인 경우에만 박스 그리기
                label = int(det[5])  # 객체의 클래스 인덱스
                box = det[:4].cpu().numpy()  # 박스 좌표

                # 박스 그리기
                color = (0, 255, 0)  # 녹색
                annotator.box_label(box, self.model.names[label], color)
                door_result = self.model.names[label]
            else:
                cv2.putText(img, "NOT DETECT DOOR", (320,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        
        img = annotator.result()

        img = cv2.resize(img, (640, 640))  # display 크기 통일

        return img, door_result