FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    unzip \
    vim \
    tmux \
    git \
    build-essential \
    libglib2.0-dev \
    ninja-build \
    curl \
    wget

ENV DEBIAN_FRONTEND=noninteractive

RUN python3 -m pip install --upgrade pip ipython ipdb

COPY . /opt/balrog/

WORKDIR /opt/balrog

RUN python3 -m pip install '.'
RUN balrog-post-install

CMD ["/bin/bash"]
