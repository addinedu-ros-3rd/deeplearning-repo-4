FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

ARG UID=
ARG USER_NAME=

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

RUN groupadd -g 999 appuser
RUN useradd -r -u 999 -g appuser appuser

ENV USER appuser

ADD . /workspace/project/

RUN apt update -y \
    && apt upgrade -y \
    && apt install -y vim lsb-release libgl1-mesa-dev libglib2.0-0 \
    && pip install ultralytics tensorflow opencv-python koreanize-matplotlib

ENV DEBIAN_FRONTEND newt