FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS builder

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

ADD . /workspace/project/

RUN apt update --no-install-recommends \
    && apt upgrade --no-install-recommends -y \
    && apt install -y sudo vim lsb-release libgl1-mesa-dev libglib2.0-0 qt5-default libxcb-xinerama0-dev \
    && pip --default-timeout=300 install ultralytics tensorflow opencv-python koreanize-matplotlib mediapipe mysql-connector-python

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