import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

model = keras.models.load_model(os.path.dirname(os.path.abspath(__file__)) +\
                                "/arm_pose_model")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

pose_type = ["None", "Left Arm Raised", "Right Arm Raised", "Both Arm Raised"]

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()

    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        try:
            # landmarker값 추출, predict
            test = []
            for i in range(11, 17):
                target_values = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]
                test.append(target_values.x)
                test.append(target_values.y)
            test = np.array(test, dtype=np.float32).reshape(1, -1)
            print(test)
            pred = model.predict(test)
            now_pose = pose_type[np.argmax(pred[0])]
        except:
            continue

        # 보기 편하게 좌우 반전
        image = cv2.flip(image, 1)
        cv2.putText(image, now_pose,
                    org=(50,50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    thickness=2,
                    color=(255,0,0))
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cam.release()
