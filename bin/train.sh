#!/bin/bash
echo "Begin to train ..."
Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0

if [ "$1" == "console" ] || [ "$1" == "debug" ]; then

    if [ "$1" == "debug" ]; then
        echo "_/_/_/_/_/_/  Start PDB Debugging...  _/_/_/_/_/_/"
        sed -i '1i\import pdb; pdb.set_trace()\n' main/train.py
    fi

    if [ "$2" != "" ]; then
        echo "User define GPU #$2"
        export CUDA_VISIBLE_DEVICES=$2
    fi

    echo "In DEBUG mode ..."
    python -m main.train \
    --name=textscanner \
    --epochs=3 \
    --debug \
    --debug_step=3 \
    --steps_per_epoch=3 \
    --batch=3 \
    --retrain=True \
    --learning_rate=0.001 \
    --train_label_dir=data/train.english \
    --validate_label_dir=data/validate.english \
    --workers=3 \
    --early_stop=10

    if [ "$1" == "debug" ]; then
        # 恢复源文件，防止git提交
        sed -i '1d' main/train.py
    fi

    exit
fi

if [ "$1" = "stop" ]; then
    echo "Stop Training!"
    ps aux|grep python|grep name=textscanner|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" != "" ]; then
    echo "User define GPU #$1"
    export CUDA_VISIBLE_DEVICES=$1
fi

echo "Production Mode ..."
echo "Using #$CUDA_VISIBLE_DEVICES GPU"

python -m main.train \
    --name=textscanner \
    --steps_per_epoch=2000 \
    --epochs=5000000 \
    --debug_step=1000 \
    --batch=32 \
    --retrain=False \
    --learning_rate=0.001 \
    --validation_batch=10 \
    --validation_steps=100 \
    --train_label_dir=data/train \
    --validate_label_dir=data/validate \
    --workers=10 \
    --early_stop=100 \
    >> ./logs/Console_GPU$CUDA_VISIBLE_DEVICES_$Date.log 2>&1
