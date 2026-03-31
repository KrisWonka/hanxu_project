FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    iproute2 \
    iputils-ping \
    libportaudio2 \
    locales \
    nano \
    net-tools \
    openssh-client \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    sudo \
    tzdata \
    vim \
    wget \
    && locale-gen en_US.UTF-8 \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID dev \
    && useradd -m -u $UID -g $GID -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/dev

RUN git config --system --add safe.directory /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG DEV_HOME=/home/dev
ENV PATH="${DEV_HOME}/.local/bin:${PATH}"

USER dev
WORKDIR /workspace

CMD ["bash"]
