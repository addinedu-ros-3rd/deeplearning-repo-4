# 🌞
## 프로젝트 간략 설명
**학원에서 사용할 수 있는 학습도우미 로봇을 만들고자 하였으며**

**로봇이 제공하는 기능은 크게 5가지로 계획하였다**

### 주요 기능
1. **Phone detect**
   - 휴대폰을 사용하는지 안 하는지 검사
2. **Door detect**
   - 문이 열려있는지 닫혀잇는지 판단
3. **Light detect**
   - 불이 켜졌는지 닫혔는지 판단 
4. **Snack detect**
   - 간식이 몇 % 있는지 판단
5. **Desk detect**
   - 책상위의 사물 판단
6. **영상 녹화 및 재생**
   - 현재 화면을 녹화 하고 재생
7. **검색 기능**
   - detect된 정보들을 검색할 수 있음
   - 요청 ID, 저장된 시간, 기능이름, 인식결과, 저장 파일 경로
    
### 통신
- 주요기능의 1~5번기능은 ROS를 사용하여 카메라로부터 나오는 /image_raw 토픽을 받아 처리

### Docker
- 배포는 Docker를 사용하여 패키징

### 데이터 베이스 (ERD)
- 데이터 베이스
  
![image](https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/146153568/cdcd0e05-f24b-4878-8a63-9a0eee9687b2)

### 실시간 시나리오
- 시스템의 동작 시나리오
  
![image](https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/146153568/761d2643-9121-4ce5-aede-a1929f72764b)

### 저장된 영상 확인 시나리오
- 로봇이 영상을 저장한 뒤 그 영상을 확인하는 시나리오
  
![image](https://github.com/addinedu-ros-3rd/deeplearning-repo-4/assets/146153568/d301ecf6-b505-4271-84bf-706379cbb01d)

## 진척도 

현재 5가지의 기능을 80% 정도 구현하였으며

GUI 제작, DB 연동, PPT 제작 작업을 진행 중에 있으며

시간이 된다면 Docker로 패키징할 예정


