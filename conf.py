import argparse,sys

'''
    define the basic configuration parameters,
    also define one command-lines argument parsing method: init_args
'''
MAX_SEQUENCE = 30        # 最大的识别汉字的长度
MASK_VALUE = 0
CHARSET = "config/charset.4100.txt" # 一级字库+标点符号+数字+二级字库中的地名/人名常用字（TianT.制作的）
INPUT_IMAGE_HEIGHT = 64  # 图像归一化的高度
INPUT_IMAGE_WIDTH = 256  # 最大的图像宽度
GRU_HIDDEN_SIZE = 256    # GRU隐含层神经元数量
FEATURE_MAP_REDUCE = 8   # 相比原始图片，feature map缩小几倍（送入bi-gru的解码器之前的feature map），目前是8，因为用的resnet50，缩小8倍
FILTER_NUM = 256         # 自定义层中的默认隐含神经元的个数

DEBUG = True

DIR_LOGS="logs"
DIR_TBOARD="logs/tboard"
DIR_MODEL="model"
DIR_CHECKPOINT="model/checkpoint"
LABLE_FORMAT="plaintext" # 标签格式：labelme，json格式的；plaintext，纯文本的

# 伐喜欢tensorflow的flags方式，使用朴素的argparse
# dislike the flags style of tensorflow, instead using argparse
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name" ,default="attention_ocr",type=str,help="")
    parser.add_argument("--train_label_dir",    default="data/train",    type=str, help="")
    parser.add_argument("--validate_label_dir", default="data/train", type=str, help="")
    parser.add_argument("--train_label_file",    default="data/train/train.txt", type=str, help="")
    parser.add_argument("--validate_label_file", default="data/train/train.txt", type=str, help="")
    parser.add_argument("--epochs" ,default=1,type=int,help="")
    parser.add_argument("--debug_mode", default=False, action='store_true', help="")
    parser.add_argument("--debug_step", default=1,type=int,help="") # 多少步骤打印注意力
    parser.add_argument("--steps_per_epoch", default=None,type=int,help="")
    parser.add_argument("--batch" , default=1,type=int,help="")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="")
    parser.add_argument("--workers",default=1,type=int,help="")
    parser.add_argument("--retrain", default=False, type=bool, help="")
    parser.add_argument("--preprocess_num",default=None,type=int,help="") # 整个数据的个数，用于调试，None就是所有样本
    parser.add_argument("--validation_steps",default=1,type=int, help="")
    parser.add_argument("--validation_batch",default=1,type=int, help="")
    parser.add_argument("--early_stop", default=1, type=int, help="")
    args = parser.parse_args()
    print("==============================")
    print(" Configurations : ")
    print("==============================")
    print(args)

    sys.modules[__name__].DEBUG = args.debug_mode

    if args.debug_mode:
        print("Running in DEBUG mode!")
        sys.modules[__name__].FILTER_NUM = 1

    return args


def init_pred_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image" ,default=1,type=str,help="")
    parser.add_argument("--model" ,default=1,type=str,help="")
    args = parser.parse_args()
    return args