import cv2
import torch
from rclpy.node import Node
from haejo_pkg.utils.ConfigUtil import get_config
from haejo_pkg.utils import Logger


log = Logger.Logger('detect_snack.py')
config = get_config()


class DetectSnack(Node):
    def __init__(self):
        super().__init__('snack_detect')

        self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=config['snack_model'], force_reload=False)

        # 모델을 추론 모드로 설정
        self.model.eval()
        
        log.info("success detect_snack model load")
    
    def detect_snack(self, img):
        
        snack_result = ""

        # 추론 수행
        results = self.model(img)
        log.info(results)

        # 각 객체에 대해 결과 확인
        for i, det in enumerate(results.xyxy[0]):
            score = float(det[4])  # 감지된 객체의 확률
            if score >= 0.5:  # 확률이 0.5 이상인 경우에만 박스 그리기
                label = int(det[5])  # 객체의 클래스 인덱스
                box = det[:4].cpu().numpy()  # 박스 좌표

                # 박스 그리기
                color = (0, 0, 255)  # 녹색
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(img, f"{self.model.names[label]}: {score:.2f}", (int(box[0]), int(box[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                snack_result = self.model.names[label]
                
        img = cv2.resize(img, (640, 640))  # display 크기 통일

        return img, snack_result