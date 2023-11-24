FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV LANG C.UTF-8

# to install with no interactions
ENV DEBIAN_FRONTEND noninteractive

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

# add all current directory files in docker
ADD . /workspace/project/

RUN apt update --no-install-recommends
RUN apt upgrade --no-install-recommends -y
RUN apt install -y sudo vim lsb-release libgl1-mesa-dev libglib2.0-0 qt5-default libxcb-xinerama0-dev
RUN pip install ultralytics tensorflow opencv-python koreanize-matplotlib mediapipe transformers timm

# reset environment default
ENV DEBIAN_FRONTEND newt

# create user
ARG USER_NAME=
ARG USER_ID=
ARG GROUP_ID=

RUN groupadd ${USER_NAME} --gid ${GROUP_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash
RUN passwd -d ${USER_NAME}

# grant sudo to user
RUN usermod -aG sudo $USER_NAME

# for using webcam: grant video to user
RUN usermod -aG video $USER_NAME

# change user
USER $USER_NAME