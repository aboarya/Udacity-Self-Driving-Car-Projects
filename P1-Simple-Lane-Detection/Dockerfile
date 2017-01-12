# Dockerfile to setup opencv 3.1.0 for use outside of Jupyter Notebook
FROM ubuntu:16.04

ENV PYTHON_VERSION 3.5
ENV NUM_CORES 4

# Install OpenCV 3.1
RUN apt-get -y update
RUN apt-get -y install python$PYTHON_VERSION-dev wget unzip \
                       build-essential cmake git pkg-config libatlas-base-dev gfortran \
                       libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev python3-tk


RUN wget https://bootstrap.pypa.io/get-pip.py && /usr/bin/python3.5 get-pip.py
RUN pip install matplotlib>=1.5.3 numpy>=1.11.2

RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv3.zip && \
    unzip -q opencv3.zip && mv /opencv-3.1.0 /opencv

RUN wget https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && mv /opencv_contrib-3.1.0 /opencv_contrib

RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON \
	-D WITH_IPP=OFF \
	-D WITH_V4L=ON \
	-D CMAKE_INSTALL_PREFIX=$(python3.5 -c "import sys; print(sys.prefix)")\
	-D PYTHON3_EXECUTABLE=$(which python3.5) \
	-D PYTHON3_INCLUDE_DIR=$(python3.5 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-D PYTHON3_PACKAGES_PATH=$(python3.5 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
	
RUN make -j$NUM_CORES
RUN make install
RUN ldconfig

ADD test_images /lane-detection/test_images
ADD main.py /lane-detection/main.py

WORKDIR /lane-detection
VOLUME ["/lane-detection"]

# Define default command.
CMD ["bash"]
