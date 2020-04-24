if [ "$1" == "" ] then;
    tboard.sh <port>
    exit
fi

nohup /root/py3/bin/tensorboard --port=$1 --logdir=./logs/tboard >/dev/null 2>&1 &
