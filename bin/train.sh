#!/bin/bash
# 参数说明：
# python -m main.train \
#    --name=attention_ocr \
#    --epochs=200 \                 # 200个epochs，但是不一定能跑完，因为由ealy stop
#    --steps_per_epoch=1000 \       # 每个epoch对应的批次数，其实应该是总样本数/批次数，但是我们的样本上百万，太慢，所以，我们只去1000个批次
#                                   # 作为一个epoch，为何要这样呢？因为只有每个epoch结束，keras才回调，包括validate、ealystop等
#    --batch=64 \
#    --learning_rate=0.001 \
#    --validation_batch=64 \
#    --retrain=True \               # 从新训练，还是从checkpoint中继续训练
#    --validation_steps=10 \        # 这个是说你测试几个批次，steps这个词不好听，应该是batchs，实际上可以算出来，共测试64x10=640个样本
#    --workers=10 \
#    --preprocess_num=100 \
#    --early_stop=10 \              # 如果10个epochs都没提高，就停了吧，大概是1万个batch

echo "Begin to train ..."

Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0


if [ "$1" == "console" ] || [ "$1" == "debug" ]; then

    if [ "$1" == "debug" ]; then
        echo "_/_/_/_/_/_/  Start PDB Debugging...  _/_/_/_/_/_/"
        sed -i '1i\import pdb; pdb.set_trace()\n' main/train.py
    fi

    echo "In DEBUG mode ..."
    #    --validation_steps=1  \
    # 测试：
    # 训练：10张训练，但是steps_per_epoch=2，batch=3，预想6张后，就会重新shuffle
    # 验证：使用sequence是不需要要validation_steps参数的，他会自己算，len(data)/batch
    #      如果你规定，那就得比它小才可以，另外还要验证，是不是把每个批次的结果做平均，还是算整体的
    export CUDA_VISIBLE_DEVICES=0 # 调试不用GPU
    python -m main.train \
    --name=textscanner \
    --epochs=1 \
    --debug_mode \
    --debug_step=1 \
    --steps_per_epoch=1 \
    --batch=3 \
    --retrain=True \
    --learning_rate=0.001 \
    --train_label_dir=data/test \
    --validate_label_dir=data/test \
    --validation_batch=1 \
    --validation_steps=1 \
    --preprocess_num=1 \
    --workers=3 \
    --early_stop=1

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


echo "Production Mode ..."
echo "Using #$CUDA_VISIBLE_DEVICES GPU"

nohup python -m main.train \
    --name=textscanner \
    --steps_per_epoch=1000 \
    --epochs=5000000 \
    --debug_step=1000 \
    --batch=32 \
    --retrain=True \
    --learning_rate=0.001 \
    --validation_batch=64 \
    --validation_steps=10 \
    --workers=10 \
    --early_stop=100 \
    >> ./logs/Attention_GPU$CUDA_VISIBLE_DEVICES_$Date.log 2>&1 &
