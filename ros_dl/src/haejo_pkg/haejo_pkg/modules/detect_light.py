import cv2
import numpy as np
from rclpy.node import Node
from tensorflow.keras.models import load_model


class Detectlight(Node):
    def __init__(self):
        super().__init__('light_detect')

        custom_model_path = '/home/soomin/ros_test/src/haejo_pkg/model/light_on_off_model.keras'
        
        self.model = load_model(custom_model_path)
        
        print("success detect_light model load")  
              

    def detect_light(self, img):

        # 전처리: 크기 조정 등 모델의 입력에 맞게 프레임을 전처리해야 함
        
        processed_frame = cv2.resize(img, (195, 195))
        processed_frame = processed_frame / 255.0  # 모델이 학습할 때 정규화되었는지 확인

        # 모델 예측
        input_array = np.expand_dims(processed_frame, axis=0)  # 모델은 배치 차원을 기대하므로 차원 확장
        predictions = self.model.predict(input_array)

        
        class_probabilities = predictions[0]
        class_index = np.argmax(class_probabilities)

        # class_labels 리스트에 "OFF"와 "ON"을 순서대로 넣어둔다고 가정합니다.
        class_labels = ["OFF", "ON"]

        # class_index를 기반으로 클래스 레이블 설정
        class_label = class_labels[class_index]

        # 확률을 퍼센트로 변환하여 문자열 구성
        class_probability_percent = class_probabilities[class_index] * 100
        class_label_text = f"Light Check: {class_label}, Probability(%): {class_probability_percent:.4f}"

        # 결과를 화면에 출력
        cv2.putText(img, class_label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 10, 0), 2)


        return img

    


