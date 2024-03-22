FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG CUDA_ARCHITECTURES=90;89;86;80;75

ENV DEBIAN_FRONTEND=noninteractive
# NOTE: Timezone is randomly set.
ENV TZ=Europe/Berlin
ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y

RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python3.9 \
    python3.9-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    libssl-dev \
    unzip \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*
RUN apt remove -y python3-distro-info


RUN rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3.9 /usr/bin/python
ENV HOME /src
WORKDIR /src
ENV SHELL /bin/bash

ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry
COPY ["./pyproject.toml", "/src/"]
COPY image2layout /src/image2layout
RUN poetry config virtualenvs.create true
RUN poetry install