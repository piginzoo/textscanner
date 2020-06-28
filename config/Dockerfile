From tensorflow/tensorflow:2.1.0-gpu-py3

MAINTAINER piginzoo

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup
ADD config/sources.list /etc/apt/sources.list
RUN apt-get update
#RUN apt-get install -y vim build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# support cv2
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev
RUN apt-get install -y vim
RUN apt-get install -y net-tools

RUN mkdir /root/.pip
ADD config/pip.conf /root/.pip
ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

# time zone
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
RUN apt install tzdata
RUN rm /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai">/etc/timezone

# we cannot find tf2.2.0 image, so we use tf2.1 to upgrade
RUN pip install tensorflow-gpu==2.2.0
RUN pip install tensorboard_plugin_profile