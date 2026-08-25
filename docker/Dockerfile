FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV TZ=Europe/Brussels
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_INSTALL_DIR=/usr/bin
ENV ZSH_THEME=bira


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget zsh curl git ansible sshpass vim build-essential sudo gosu ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -t ${ZSH_THEME}

RUN curl -LsSf https://astral.sh/uv/0.7.21/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh

WORKDIR /home/workspace
COPY src/ /home/workspace/src/
COPY pyproject.toml uv.lock README.md /home/workspace/
RUN uv pip install --no-cache-dir --system -e '.'