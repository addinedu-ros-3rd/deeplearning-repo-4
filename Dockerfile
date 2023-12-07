FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

ADD . /workspace/

# install python and pip libraries for deep learning
RUN apt update --no-install-recommends \
    && apt upgrade --no-install-recommends -y \
    && apt install -y sudo vim lsb-release libgl1-mesa-dev libgl1-mesa-glx libglib2.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
        libxcb-xinerama0-dev python3.10 python3-pip \
    && apt install -y make libgtk2.0-dev pkg-config g++ libxcb*-dev libfontconfig1-dev libxkbcommon-x11-dev libgtk-3-dev \
    # install ros2 humble & install ros libs & build
    && apt install -y software-properties-common \
    && add-apt-repository universe \
    && apt update && sudo apt install curl -y \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update && yes | apt install ros-humble-desktop python3-colcon-common-extensions \
    && echo 'export QT_QPA_PLATFROM="xcb"' > /etc/environment

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

WORKDIR /workspace

RUN pip --default-timeout=300 install -r ./requirements.txt