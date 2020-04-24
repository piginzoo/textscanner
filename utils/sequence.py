from utils import image_utils, label_utils
import logging,math
import numpy as np

from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

import time

logger = logging.getLogger("SequenceData")


import matplotlib
def show_image(img):
    matplotlib.pyplot.imshow(img)


# 自定义的数据加载器
# 几个细节：
# - 不用用多进程方式加载，不知道为何总是卡住，改成multiprocess=False,即使用多线程就好了,参考：https://stackoverflow.com/questions/54620551/confusion-about-multiprocessing-and-workers-in-keras-fit-generator-with-window
# - on_epoch_end确实是所有的样本都轮完了，才回调一次，而，steps_per_epoch改变的是多久callback回调一次，这个可以调的更小一些，两者没关系
# - (2020.3.27)on_epoch_end回调是所有的样本都完成，和steps_per_epoch无关，
#              比如10000张样本，我batch设成100，但是steps_per_epoch设成50，
#              那么，不是5000张就会调用on_epoch_end，还是要等到10000张都轮完了，才会回调这个方法
class SequenceData(Sequence):
    def __init__(self, name,label_file, charset_file,conf,args,batch_size=32):
        self.conf = conf
        self.name = name
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = label_utils.get_charset(charset_file)
        self.initialize(conf,args)
        self.start_time = time.time()

    # 返回长度，我理解是一个epoch中的总步数
    # 'Denotes the number of batches per epoch'
    def __len__(self):
        # logger.debug("[%s],__len__:%d",self.name,int(math.ceil(len(self.data_list) / self.batch_size)))
        return int(math.ceil(len(self.data_list) / self.batch_size))


    def load_image_label(self,batch_data_list):
        # logger.debug("[%s]加载标签:%r",self.name, batch_data_list)
        images_labelids = label_utils.process_lines(self.charsets,batch_data_list)

        # print(self.name,"Sequence PID:", os.getpid(),",idx=",idx)
        # unzip的结果是 [(1,2,3),(a,b,c)]，注意，里面是个tuple，而不是list，所以用的时候还要list()转化一下
        # zip(*xxx）操作是为了解开[(a,b),(a,b),(a,b)]=>[a,a,a][b,b,b]
        image_names, label_ids = list(zip(*images_labelids))

        # 读取图片，高度调整为32，宽度用黑色padding
        images = image_utils.read_and_resize_image(list(image_names),self.conf)

        # labels是[nparray([<3773>],[<3773>],[<3773>]),...]，是一个数组，里面是不定长的3370维度的向量,(N,3770),如： (18, 3861)
        labels = list(label_ids)
        labels = pad_sequences(labels,maxlen=self.conf.MAX_SEQUENCE,padding="post",value=0)
        labels = to_categorical(labels,num_classes=len(self.charsets))        #to_categorical之后的shape： [N,time_sequence(字符串长度),3773]

        return images,labels

    # 即通过索引获取a[0],a[1]这种,idx是被shuffle后的索引，你获取数据的时候，需要[idx * self.batch_size : (idx + 1) * self.batch_size]
    # 2019.12.30,piginzoo，
    def __getitem__(self, idx):
        start_time = time.time()
        logger.debug("[%s]加载批次index:%r",self.name,idx)
        batch_data_list = self.data_list[ idx * self.batch_size : (idx + 1) * self.batch_size]

        images,labels = self.load_image_label(batch_data_list)

        # logger.debug("[%s]进程[%d],加载一个批次数据，idx[%d],耗时[%f]",
        #             self.name,
        #             os.getpid(),
        #             idx,
        #             time.time()-start_time)
        # 识别结果是STX,A,B,C,D,ETX，seq2seq的decoder输入和输出要错开1个字符
        # labels[:,:-1,:]  STX,A,B,C,D  decoder输入标签
        # labels[:,1: ,:]  A,B,C,D,ETX  decoder验证标签
        # logger.debug("加载批次数据：%r",images.shape)
        # logger.debug("Decoder输入：%r", labels[:,:-1,:])
        # logger.debug("Decoder标签：%r", labels[:,1:,:])
        return [images,labels[:,:-1,:]],labels[:,1:,:]

    # 一次epoch后，重新shuffle一下样本
    def on_epoch_end(self):
        np.random.shuffle(self.data_list)
        duration = time.time() - self.start_time
        self.start_time = time.time()
        logger.debug("[%s]本次Epoch结束，耗时[%d]秒，重新shuffle数据",self.name,duration)

    # 初始加载样本：即每一个文件的路径和对应的识别文字
    # 额外做两件事：
    # 1、check每一个图片文件是否存在
    # 2、看识别文字的字不在字表中，剔除这个样本
    def initialize(self,conf,args):
        logger.info("[%s]开始加载样本和标注",self.name)
        start_time = time.time()
        self.data_list = label_utils.read_data_file(self.label_file,args.preprocess_num)
        logger.info("[%s]加载了样本:[%d]个,耗时[%d]秒", self.name, len(self.data_list),(time.time() - start_time))
