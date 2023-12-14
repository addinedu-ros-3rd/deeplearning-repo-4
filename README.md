# 🌞 딥러닝 학습도우미
## 1. 프로젝트 소개
### 학습 방해 요소를 확인하는 딥러닝 모델 개발
<p align=center width="100%">
  <img src="/image/door.gif" height="300" width="22%" style="float:left">
  <img src="/image/snack.gif" height="300" width="22%" style="float:left">
  <img src="/image/desk.gif" height="300" width="22%" style="float:left">
  <img src="/image/phone.gif" height="300" width="22%" style="float:left">
</p>

- 객체 인식
  - 문 개폐 여부: open / closed
  - 간식 상자 채워진 비율: 0%, 25%, 50%, 75%, 100%
  - 책상 위 사물: 쓰레기(비닐, 플라스틱, 종이, 휴지, 커피컵), 그외(쿠키, 커피)
- 분류
  - 자세: 휴대폰 사용 중 / 집중하고 있음
  - 밝기: 밝음 / 어두움
 
- 프로젝트 기간: 2023.11.17 ~ 12.15

### 기술 스택
|   |   |
|---|---|
|개발환경|![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white) ![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white) ![RDS](https://img.shields.io/badge/AWS%20RDS-527FFF?style=for-the-badge&logo=Amazon%20RDS&logoColor=white) ![S3](https://img.shields.io/badge/AWS%20S3-569A31?style=for-the-badge&logo=Amazon%20S3&logoColor=white) ![Qt](https://img.shields.io/badge/Qt-41CD52?style=for-the-badge&logo=Qt&logoColor=white)||
|기술|![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white) ![Tensorflow](https://img.shields.io/badge/Tensorflow-FF6F00?style=for-the-badge&logo=Tensorflow&logoColor=white) ![ROS2](https://img.shields.io/badge/ROS2-22314E?style=for-the-badge&logo=ROS&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white) ![Mysql](https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white)|
|커뮤니케이션|![Jira](https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=Jira&logoColor=white) ![Confluence](https://img.shields.io/badge/Confluence-172B4D?style=for-the-badge&logo=Confluence&logoColor=white) ![Slack](https://img.shields.io/badge/slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)|

### 주요 기능
![Screenshot from 2023-12-14 20-16-41](https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/bcdc6e39-b7c3-45e0-b3db-93ad594d28a4)

## 2. 개발 문서
### 시스템 구성도
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/afc45aae-57bc-4871-a21b-c6758ac1c98d" width="90%" style="float:left">
</p>

<br><br>

### 실시간 시나리오
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/1638e49e-4856-4cb7-81df-ae5777c056c4" width="90%" style="float:left">
</p>

<br><br>

### 저장된 영상 확인 시나리오
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/4331709c-9ba8-4cc9-895f-cdb1d0a18c7b" width="90%" style="float:left">
</p>

<br><br>

### 객체 인식 데이터 구조
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/146153568/cdcd0e05-f24b-4878-8a63-9a0eee9687b2" width="70%" style="float:left">
</p>

## 3. 실행 방법
### 3-1. 딥러닝 모델: 도커 환경
- deploy 디렉토리에서
  ```
  docker compose up -d
  ```
  로 이미지 빌드 후 실행 시, 사용하는 PC의 웹캠과 디스플레이에 연동하여 모델 성능 확인이 가능합니다.

### 3-2. UI/DB/S3/ROS2 연동 프로그램 실행: 로컬 PC
0. 촬영한 영상 저장과 조회를 위해서는 AWS S3 bucket과 IAM 생성이 필요합니다. 이 설정이 없을 경우 실시간 인식만 가능합니다.
1. ros_cam 프로젝트(https://github.com/ros-drivers/usb_cam) 와 직접 작성한 ros_dl 프로젝트를 동시에 실행해야 합니다.
  - ros_cam을 실행한 PC와 USB로 연결된 카메라의 /image_raw 토픽을 받아올 수 있는(같은 ROS_DOMAIN_ID를 가진) PC에서 ros_dl 프로젝트를 실행합니다.
  - 하나의 PC에서 ros_cam과 ros_dl을 동시에 실행할 수도 있습니다.
2. 실행에 앞서, ros2 humble과 qt5, requirements의 pip 라이브러리가 설치되어 있어야 합니다.
  - ros2 humble 설치: https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html
  - qt5 설치 (ubuntu 22.04 기준)
  ```
  sudo apt-get update
  sudo apt-get upgrade
  sudo apt-get install build-essential make qt5-default qtcreator qtdeclarative5-dev libqt5serialport5-dev 
  ```
  - python 라이브러리 설치
  ```
  pip install -r requirements.txt
  ```
3. 프로젝트의 ConfigUtil.py 파일에서 config.ini 파일이 있는 경로 수정이 필요합니다.
  ```
  configParser.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')
  config = configParser['yun']
  ```
4. create_and_init.sql 파일로 DB의 테이블과 SP를 생성하고 모듈 테이블의 데이터를 추가합니다.
   ```
   source create_and_init.sql
   ```
5. config.ini는 다음 형태로 작성 필요합니다.
  ```
  [dev]
  host = DB Host
  port = DB port
  user = DB user
  password = DB password
  database = DB database name

  GUI = /workspace/ros_dl/src/haejo_pkg/haejo_pkg/haejo.ui
  phone_yolo_model = /workspace/ros_dl/src/haejo_pkg/model/yolov5su.pt
  phone_lstm_model = /workspace/ros_dl/src/haejo_pkg/model/yolo_state_dict.pt

  desk_model = /workspace/ros_dl/src/haejo_pkg/model/desk_best.pt
  door_model = /workspace/ros_dl/src/haejo_pkg/model/door_best.pt
  light_model = /workspace/ros_dl/src/haejo_pkg/model/light_on_off_model.keras
  snack_model = /workspace/ros_dl/src/haejo_pkg/model/snack_best.pt

  video_dir = /workspace/record/
  ```
  - keras를 제외한 모든 모델은 현재 git 프로젝트에서 ros_dl/src/haejo_pkg/model 경로 하위에 포함되어 있습니다.
  - keras 모델은 구글 드라이브에서 받아주세요: https://drive.google.com/file/d/18llPFcgIQBvfEDXJx_6Zr1BPR_P9KzbJ/view?usp=sharing
6. ros2 실행
    ```
    source /opt/ros/humble/setup.bash
    ```
7. ros2 프로젝트 빌드
  - ros_cam 경로
    ```
    rosdep init
    rosdep update
    rosdep install --from-paths src --ignore-src -y
    colcon build
    source ./install/local_setup.bash
    ```
  - ros_dl 경로
    ```
    colcon build
    source ./install/local_setup.bash
    ```
8. ros2 프로젝트 실행
  - 카메라 토픽 발행
     ```
     ros2 run usb_cam usb_cam_node_exe
     ```
  - 토픽 구독 및 UI 실행
     ```
     ros2 run haejo_pkg deep_learning
     ```

## 4. 팀원 소개 및 역할
|구분|이름|역할|
|---|---|---|
|팀장|이충한|PM, PyQt GUI 개발, 데이터 라벨링|
|팀원|강병철|CNN/VGG16 전이학습, YOLOv5 활용|
|팀원|오윤|시스템 구성도 및 시나리오 작성, YOLOv5 활용, DB 설계, AWS S3 연동, Query/SP 작성, Dockerfile/compose.yaml 작성|
|팀원|이수민|LSTM/YOLOv8 활용, ROS2, mediapipe, PyQt|

## 5. 회고
- 기술적으로 만족한 점
  - 도커로 개발 환경 구축
  - ROS2 통신 연동
  - 직접 구축한 데이터셋으로 모델 학습 진행
- 아쉬운 점
  - 다양한 딥러닝 모델로 학습하지 못했고, 모델 경량화가 부족하여 실시간 인식 시 PC 자원에 따라 화면 버퍼링 발생
  - 도커 운영 환경을 구축하고자 했으나, UI 코드를 분리하지 못해 Qt 로컬 환경 의존성 이슈 발생
  - 움직이는 로봇에 연동된 카메라를 상정하여 ESP-32 cam 사용 및 Wifi/TCP/UDP 통신을 고려했으나, 구현하지 못하고 USB CAM과 ROS 통신만 사용
