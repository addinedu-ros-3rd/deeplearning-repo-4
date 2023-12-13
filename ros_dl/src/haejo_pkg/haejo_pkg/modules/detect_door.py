import cv2
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator

class Detectdoor(Node):
    def __init__(self):
        super().__init__('door_detect')
       
        custom_model_path = '/home/soomin/ros_test/src/haejo_pkg/model/detect_door.pt'

        # 모델을 로드할 때 허브 모듈을 강제로 다시 로드하도록 설정
        self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=custom_model_path, force_reload=True)

        # 모델을 추론 모드로 설정
        self.model.eval()
        
        print("success detect_door model load")        
    
    def detect_door(self, img):

        # 추론 수행
        results = self.model(img)
        print(results)

        if results.xyxy[0].tolist() == []:
            cv2.putText(img, "NOT DETECT DOOR", (320,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        # 각 객체에 대해 결과 확인

        annotator = Annotator(img)
        
        for i, det in enumerate(results.xyxy[0]):
            score = float(det[4])  # 감지된 객체의 확률
            print(score)
            if score >= 0.5:  # 확률이 0.5 이상인 경우에만 박스 그리기
                label = int(det[5])  # 객체의 클래스 인덱스
                box = det[:4].cpu().numpy()  # 박스 좌표

                # 박스 그리기
                color = (0, 255, 0)  # 녹색
                annotator.box_label(box, self.model.names[label], color)
            else:
                cv2.putText(img, "NOT DETECT DOOR", (320,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        
        img = annotator.result()
        # 결과 표시

        return img        

    


