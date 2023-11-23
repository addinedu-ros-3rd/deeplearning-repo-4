import cv2
import torch.nn as nn
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

mp_pose = mp.solutions.pose
attention_dot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
draw_line = [[0, 4], [0, 1], [4, 5], [1, 2], [5, 6], [2, 3], [6, 8], [3, 7], [9, 10]]

# def show_skeleton(video_path , interval, attention_dot, draw_line):
#     xy_list_list, xy_list_list_flip = [], []
#     cv2.destroyAllWindows()
#     pose = mp_pose.Pose(static_image_mode = True, model_complexity = 1, enable_segmentation = False, min_detection_confidence = 0.3)
#     cap = cv2.VideoCapture(video_path)
    
#     if cap.isOpened():
#         cnt = 0
#         while True:
#             ret, img = cap.read()
#             if cnt == interval and ret == True:
#                 cnt = 0
#                 xy_list, xy_list_flip = [], []
#                 img = cv2.resize(img, (640,  640))
#                 results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 if not results.pose_landmarks: continue
#                 idx = 0
#                 draw_line_dic = {}
#                 for x_and_y in results.pose_landmarks.landmark:
#                     if idx in attention_dot:
#                         xy_list.append(x_and_y.x)
#                         xy_list.append(x_and_y.y)
#                         xy_list_flip.append(1 - x_and_y.x)
#                         xy_list_flip.append(x_and_y.y)
#                         x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
#                         draw_line_dic[idx] = [x, y]
#                     idx += 1
#                 xy_list_list.append(xy_list)
#                 xy_list_list_flip.append(xy_list_flip)
#                 for line in draw_line:
#                     x1, y1 = draw_line_dic[line[0]][0], draw_line_dic[line[0]][1]
#                     x2, y2 = draw_line_dic[line[1]][0], draw_line_dic[line[1]][1]
#                     img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
#                 cv2.imshow('Landmark Image', img)
#                 if cv2.waitKey(1) == ord('q'):
#                     break
#             elif ret == False: break
#             cnt += 1
#     cap.release()
#     cv2.destroyAllWindows()
    
#     return xy_list_list + xy_list_list_flip


class MyDataset(Dataset):
    def __init__(self, seq_list):
        self.X = []
        self.y = []
        for dic in seq_list:
            self.y.append(dic['key'])
            self.X.append(dic['value'])
        
    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))
    
    def __len__(self):
        return len(self.X)


class skeleton_LSTM(nn.Module):
    def __init__(self):
        super(skeleton_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size= 22, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x
    
## load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("../model/aa.pt", map_location=device)
model.eval()

interval = 1
length = 20
img_list = []

# cv2.destroyAllWindows()
cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
if cap.isOpened():
    cnt = 0
    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (640, 640))
            if cnt == interval:
                img_list.append(img)
                cnt = 0
            dataset = []
            status = 'None'
            pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                enable_segmentation = False, min_detection_confidence=0.3)

            xy_list_list = []
            for img in img_list:
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks: continue
                xy_list = []
                idx = 0
                draw_line_dic = {}
                for x_and_y in results.pose_landmarks.landmark:
                    if idx in attention_dot:
                        xy_list.append(x_and_y.x)
                        xy_list.append(x_and_y.y)
                        x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
                        draw_line_dic[idx] = [x, y]
                    idx += 1
                xy_list_list.append(xy_list)
                for line in draw_line:
                    x1, y1 = draw_line_dic[line[0]][0], draw_line_dic[line[0]][1]
                    x2, y2 = draw_line_dic[line[1]][0], draw_line_dic[line[1]][1]
                    img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                if len(xy_list_list) == length:
                    dataset = []
                    dataset.append({'key' : 0, 'value' : xy_list_list})
                    dataset = MyDataset(dataset)
                    dataset = DataLoader(dataset)
                    xy_list_list = []
                    for data, label in dataset:
                        data = data.to(device)
                        with torch.no_grad():
                            result = model(data)
                            _, out = torch.max(result, 1)
                            if out.item() == 0: status = 'Phoneing'
                            else: status = 'Working'
                cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            cv2.imshow("pose_estimation", img)
            
            if cv2.waitKey(1) == ord('q'):
                break
            cnt += 1
        else:
            break
cap.release()
cv2.destroyAllWindows()


