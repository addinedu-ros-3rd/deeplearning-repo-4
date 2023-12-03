FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

ADD . /workspace/project/

# install python and pip libraries for deep learning
RUN apt update --no-install-recommends \
    && apt upgrade --no-install-recommends -y \
    && apt install -y sudo vim lsb-release libgl1-mesa-dev libglib2.0-0 qtbase5-dev qt5-qmake libxcb-xinerama0-dev python3.10 python3-pip git \
    && pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 \
    && pip --default-timeout=300 install ultralytics tensorflow opencv-python mediapipe mysql-connector-python \
    # install ros2 humble
    && apt install -y software-properties-common \
    && add-apt-repository universe \
    && apt update && sudo apt install curl -y \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update && yes | apt install ros-humble-desktop

# reset environment default
ENV DEBIAN_FRONTEND newt

# create user
ARG USER_NAME=
ARG USER_ID=
ARG GROUP_ID=

RUN groupadd ${USER_NAME} --gid ${GROUP_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash\
    && passwd -d ${USER_NAME} \
    && usermod -aG sudo $USER_NAME \
    && usermod -aG video $USER_NAME

# change user
USER $USER_NAME