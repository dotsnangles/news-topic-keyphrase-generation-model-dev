FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common tzdata
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python-openssl git vim less python3-venv
RUN apt-get install -y \
    g++ openjdk-8-jdk
RUN apt install locales && locale-gen en_US.UTF-8 && dpkg-reconfigure locales

ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

RUN curl https://pyenv.run | bash
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

ARG PYTHON_VERSION="3.9.16"
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION
RUN $(pyenv which python3.9) -m pip install -U pip setuptools
RUN $(pyenv which python3.9) -m pip install -U poetry
RUN $(pyenv which python3.9) -m poetry config virtualenvs.in-project true
RUN pyenv global system


### in host terminal
### run this command where the dockerfile is to build a docker image
# docker build -t jeonghyeon:1.0.0 .
### run this command inside your workspace to launch a container
# docker run -itd --gpus=all --volume=$(pwd):/workspace --shm-size=64gb --name=jeonghyeon jeonghyeon:1.0.0

### in vsc command palette
# Dev Containers: Attach to Running Container

### useful commands
# docker exec -it jeonghyeon bin/bash

