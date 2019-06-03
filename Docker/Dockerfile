FROM ubuntu:latest as build

ARG OPENCV_VERSION="3.4.6"

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake build-essential ccache git libgtk2.0-dev \
    pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng++-dev curl ca-certificates && \
    cd /tmp && \
    curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -o ${OPENCV_VERSION}.tar.gz && \
    tar xvzf ${OPENCV_VERSION}.tar.gz && \
    cd /tmp/opencv-${OPENCV_VERSION} && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/build .. && \
    make -j4 && \
    make install && \
    ldconfig && \
    cd /tmp && \
    rm -rf opencv-${OPENCV_VERSION}

FROM ubuntu:latest

COPY --from=build /tmp/build /usr/local/

RUN apt-get update && apt-get -y --no-install-recommends install cmake build-essential libjpeg-dev libpng++-dev

WORKDIR /mnt/workspace/MC