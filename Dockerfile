# set the base image
FROM tensorflow/tensorflow:2.4.1-gpu

RUN echo 'Starting Docker build'

# update sources list install dependencies
RUN echo 'Installing dependencies'
RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y wget unzip cmake
RUN apt-get install -y gstreamer1.0-python3-plugin-loader \
					   libgstreamer1.0-dev \
					   libgstreamer-plugins-base1.0-dev

RUN echo 'Upgrading pip'
RUN python3 -m pip install --upgrade pip

# creating user and directory structure
RUN echo 'Creating user directory'
RUN mkdir -p /home/crc/
RUN groupadd -r crc && useradd --no-log-init -r -d /home/crc/ -g crc crc && adduser crc sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown crc:crc /home/crc

WORKDIR /home/crc/
USER crc

# install opencv
RUN echo 'Installing OpenCV'
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && unzip opencv.zip
RUN cd opencv-4.1.0 && mkdir build && cd build \
&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_EXAMPLES=OFF .. \
&& sudo make -j$(nproc) install \
&& sudo ldconfig

# copy project into docker
RUN echo 'Creating and Copy project into Docker'
RUN mkdir -p reward-faces
RUN cd reward-faces
COPY ./ ./

# install python requirements
RUN echo 'Installing requirements'
RUN python3 -m pip install --user -r docker-requirements.txt

RUN echo 'Docker build complete'

# set environment variable
ENV LIBGL_ALWAYS_INDIRECT=1
CMD python3 -u process.py
