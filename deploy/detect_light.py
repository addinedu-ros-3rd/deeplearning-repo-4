import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# 딥 러닝 모델 로드
model = load_model('./model/light_on_off_model.keras')  # 모델 파일의 경로를 지정

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    cv2.putText(frame, "light ON, OFF Check", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # 전처리: 크기 조정 등 모델의 입력에 맞게 프레임을 전처리해야 함
    # 이 부분은 모델이 학습된 입력 형식에 맞게 적절히 수정해주세요.
    processed_frame = cv2.resize(frame, (195, 195))
    processed_frame = processed_frame / 255.0  # 모델이 학습할 때 정규화되었는지 확인

    # 모델 예측
    input_array = np.expand_dims(processed_frame, axis=0)  # 모델은 배치 차원을 기대하므로 차원 확장
    predictions = model.predict(input_array)

    # 예측 결과
    # 이 부분은 예측 결과를 어떻게 처리할지에 따라 수정해야 합니다.
    # 여기서는 간단하게 클래스별로 확률을 출력합니다.

    # class_probabilities = predictions[0]
    # class_index = np.argmax(class_probabilities)
    # class_label = f"Light Check: {class_index}, Probability(%): {class_probabilities[class_index]*100:.4f}"

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
    cv2.putText(frame, class_label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 10, 0), 2)



    # 화면에 프레임 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()