import torch
from haejo_pkg.yolov5.utils.plots import Annotator, colors
import os

# avoid finding GPU and force to use CPU
os.environ["CUDA_VISIBLE_DEVICES"]=""

def run(weights, img):
    model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=weights, force_reload=True)
    model.eval()

    results = model(img, )

    # Annotator 객체 생성
    annotator = Annotator(img, line_width=3, example=str(model.names))

    # 각 객체에 대해 결과 확인 및 상자 그리기
    for i, det in enumerate(results.xyxy[0]):
        score = float(det[4])  # 감지된 객체의 확률
        if score >= 0.2:  # 확률이 0.2 이상인 경우에만 박스 그리기
            label = int(det[5])  # 객체의 클래스 인덱스
            box = det[:4].cpu().numpy()  # 박스 좌표
            annotator.box_label(box, f"{model.names[label]} {score:.2f}", color=colors(label, True))

    img = annotator.result()
    
    return img