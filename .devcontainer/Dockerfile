FROM mcr.microsoft.com/devcontainers/base:debian-12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        pkg-config \
        libgmp-dev \
        libmpfr-dev \
        gdb \
        valgrind \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
