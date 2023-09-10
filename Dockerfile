FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as base
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get -y install git

RUN pip install torch==1.2.0
RUN pip install torchvision==0.4.0

RUN pip install psutil
RUN pip install pytz
RUN pip install tqdm
RUN pip install atari_py

RUN git clone https://github.com/NVIDIA/apex
RUN pip install --upgrade pip
RUN cd apex && pip install -v --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN apt-get install -y libsm6 libxrender-dev
RUN pip install opencv-python

RUN pip install cython
RUN apt-get install zlib1g-dev
RUN git clone --recursive https://github.com/NVLabs/cule
RUN cd cule && python setup.py install