import os
import time
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

cam = cv2.VideoCapture(0)

f = open("no_arm_raise.csv", "w")
f.write("left_shoulder_x,left_shoulder_y,"+
         "right_shoulder_x,right_shoulder_y," +
         "left_elbow_x,left_elbow_y," +
         "right_elbow_x,right_elbow_y," +
         "left_wrist_x,left_wrist_y," +
         "right_wrist_x,right_wrist_y\n")


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
        
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        tmp = ""
        tmp += str(results.pose_landmarks.landmark[11].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[11].y)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[12].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[12].y)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[13].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[13].y)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[14].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[14].y)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[15].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[15].y)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[16].x)
        tmp += ","
        tmp += str(results.pose_landmarks.landmark[16].y)
        tmp += "\n"

        f.write(tmp)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    time.sleep(0.3)

cam.release()
