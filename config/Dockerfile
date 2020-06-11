From tensorflow/tensorflow:1.14.0-gpu-py3

MAINTAINER piginzoo

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup
ADD config/sources.list /etc/apt/sources.list

RUN apt-get update
RUN apt-get install -y vim

RUN mkdir /root/.pip
ADD config/pip.conf /root/.pip
ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
