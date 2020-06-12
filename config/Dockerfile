From tensorflow/tensorflow:2.1.0-gpu-py3

MAINTAINER piginzoo

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup
ADD config/sources.list /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y vim build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN mkdir /root/.pip
ADD config/pip.conf /root/.pip
ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
