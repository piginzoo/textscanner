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

# time zone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# we cannot find tf2.2.0 image, so we use tf2.1 to upgrade
# RUN pip install tensorflow-gpu==2.2.0
# RUN pip install tensorboard_plugin_profile
