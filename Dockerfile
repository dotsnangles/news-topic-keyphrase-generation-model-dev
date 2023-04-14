FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common tzdata
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y \
    && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python-openssl git vim less python3-venv

ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.3.1

ARG PYTHON_VERSION="3.9.16"

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc &&\
    echo 'eval "$(pyenv virtualenv-init -)"' >>  ~/.bashrc

RUN eval "$(pyenv init --path)"
RUN eval "$(pyenv virtualenv-init -)"

# USER ai
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

COPY pyproject.toml ./

RUN pyenv local $PYTHON_VERSION

RUN $(pyenv which python3.9) -m pip install -U pip setuptools &&\
    $(pyenv which python3.9) -m pip install -U poetry &&\
    $(pyenv which python3.9) -m pip install -U poetry update &&\
    $(pyenv which python3.9) -m pip install -U poetry shell