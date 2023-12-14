# 🌞
## 프로젝트 간략 설명
**학원에서 사용할 수 있는 학습도우미 로봇**
<p align=center width="100%">
  <img src="https://haejo.s3.ap-northeast-2.amazonaws.com/snack.gif" height="300" width="20%" style="float:left">
  <img src="https://haejo.s3.ap-northeast-2.amazonaws.com/door.gif" height="300" width="20%" style="float:left">
  <img src="https://haejo.s3.ap-northeast-2.amazonaws.com/desk.gif" height="300" width="20%" style="float:left">
  <img src="https://haejo.s3.ap-northeast-2.amazonaws.com/vllo.gif" height="300" width="20%" style="float:left">
</p>

### 주요 기능
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/027b1c49-424b-4bc9-9b2c-d049aae24012" width="90%" style="float:left">
</p>

### 소프트웨어 구성도
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/3a5e83db-e780-4cb0-bd10-0faad1960bb5" width="80%" style="float:left">
</p>

### 통신
- 주요기능의 1~5번기능은 ROS를 사용하여 카메라로부터 나오는 /image_raw 토픽을 받아 처리
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/45cb0a28-8b2e-4171-8a51-e457bed232cb" width="80%" style="float:left">
</p>

### 객체 인식 데이터 구조
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/146153568/cdcd0e05-f24b-4878-8a63-9a0eee9687b2" width="70%" style="float:left">
</p>

### 실시간 시나리오
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/ed9fbd52-8ae4-43b9-bcf8-ae23da74032b" width="90%" style="float:left">
</p>

### 저장된 영상 확인 시나리오
<p align="center">
  <img src="https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/86283716/a3db74d7-36f3-4eb2-a02f-898e199af59d" width="90%" style="float:left">
</p>


### Docker 이미지 배포
- dockerfile이 있는 경로에서
```docker compose up -d```로 실행
