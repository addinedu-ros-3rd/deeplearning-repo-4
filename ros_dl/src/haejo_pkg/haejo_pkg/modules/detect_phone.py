import cv2
import numpy as np
import torch.nn as nn
import torch
import mediapipe as mp

from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from rclpy.node import Node
from cv_bridge import CvBridge
from haejo_pkg.utils.ConfigUtil import get_config
from haejo_pkg.utils import Logger


log = Logger.Logger('modules_detect_phone.log')
config = get_config()


mp_pose = mp.solutions.pose
mp_pose_pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                            enable_segmentation = False, min_detection_confidence=0.7)
xy_list_list = []

attention_dot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
draw_line = [[0, 4], [0, 1], [4, 5], [1, 2], [5, 6], [2, 3], [6, 8], [3, 7], [9, 10]]

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
        self.lstm1 = nn.LSTM(input_size= 26, hidden_size=128, num_layers=1, batch_first=True)
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


class DetectPhone(Node):
    def __init__(self):
        super().__init__('phone_detect')

        self.bridge = CvBridge()
        self.yolo = YOLO(config['phone_yolo_model'])

        self.labels = self.yolo.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.labels))] 

        self.model = skeleton_LSTM()
        self.model.load_state_dict(torch.load(config['phone_lstm_model'], map_location="cuda"))
        self.model.eval()
        log.info("success detect_phone model load") 

    
    def pose_estimation(self, img):
        
        global xy_list_list

        length = 20
        dataset = []
        status = 'None'
        
        img = cv2.resize(img, (640, 640))

        results = mp_pose_pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks: 
            pass    

        else:
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

            
            phone_results = self.yolo(img, stream= True)

            for r in phone_results:
                
                annotator = Annotator(img)

                boxes = r.boxes
                
                for idx, box in enumerate(boxes):
                    b = box.xyxy[0]
                    c = box.cls
                
                    if int(c) == 67:
                        
                        color = self.colors[int(c)]
                        annotator.box_label(b, self.yolo.names[int(c)], color)

                        b_list = b.tolist()
                        
                        for idx in range(4):
                            xy_list.append(b_list[idx]/640.0)
                        break
                        
                    else:
                        if (len(boxes) - 1 == idx):
                            for _ in range(4):
                                xy_list.append(0.0)
                        else:
                            continue
                            
            img = annotator.result()

            if len(xy_list_list) == length:
                
                dataset = []
                dataset.append({'key' : 0, 'value' : xy_list_list})
                dataset = MyDataset(dataset)
                dataset = DataLoader(dataset)
                xy_list_list = []

                for data, label in dataset:
                    data = data.to("cuda")
                    
                    with torch.no_grad():
                        self.model = self.model.to("cuda")
                        result = self.model(data)
                        _, out = torch.max(result, 1)
                        
                        print(out.item())
                        if out.item() == 0: 
                            status = 'phone'
                            
                        else: 
                            status = 'work'

        return img, status