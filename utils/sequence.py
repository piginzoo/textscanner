from utils import image_utils, label_utils
import logging,math
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import time,cv2
from label.label import ImageLabel
from label.label_maker import LabelGenerater

logger = logging.getLogger("SequenceData")


class SequenceData(Sequence):
    def __init__(self, name, label_dir, label_file, charsets, conf, args, batch_size=32):
        self.conf = conf
        self.label_dir = label_dir
        self.name = name
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = charsets
        self.initialize(args)
        self.start_time = time.time()
        target_image_shape = (conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH)
        self.label_generator = LabelGenerater(conf.MAX_SEQUENCE,target_image_shape,charsets)

    def __len__(self):
        return int(math.ceil(len(self.data_list) / self.batch_size))

    def load_image_label(self,batch_data_list):

        images = []
        batch_cs = []
        batch_om = []
        batch_lm = []
        for image_path,json_path in batch_data_list:
            image = cv2.imread(image_path)

            with open(json_path,encoding="utf-8") as f:
                json = f.read()
            il = ImageLabel(image,json)

            character_segment, order_maps, localization_map = self.label_generator.process(il)
            character_segment = to_categorical(character_segment, num_classes=len(self.charsets)+1)

            batch_cs.append(character_segment)
            batch_om.append(order_maps)
            batch_lm.append(localization_map)

            # TODO 需要和ImageLabel的内容统一考虑
            image = cv2.resize(image, (self.conf.INPUT_IMAGE_WIDTH, self.conf.INPUT_IMAGE_HEIGHT))
            image = image/ 255.0 # TODO 要变成 float，否则报错
            images.append(image)
        return np.array(images),[np.array(batch_cs),np.array(batch_om),np.array(batch_lm)]
        # {
        #     'charactor_segmantation':np.array(batch_cs),
        #     'order_map':np.array(batch_om),
        #     'localization_map':np.array(batch_lm)
        # }

    def __getitem__(self, idx):
        # logger.debug("[%s] load index:%r",self.name,idx)
        batch_data_list = self.data_list[ idx * self.batch_size : (idx + 1) * self.batch_size]
        images,labels = self.load_image_label(batch_data_list)
        return images,labels

    def on_epoch_end(self):
        np.random.shuffle(self.data_list)
        duration = time.time() - self.start_time
        self.start_time = time.time()
        logger.debug("[%s] Epoch done, elapsed time[%d]s，re-shuffle",self.name,duration)

    def initialize(self,args):
        logger.info("[%s]begin to load image/labels",self.name)
        start_time = time.time()
        self.data_list = label_utils.load_labels(self.label_dir,args.preprocess_num)
        logger.info("[%s]loaded [%d] labels,elapsed time [%d]s", self.name, len(self.data_list),(time.time() - start_time))
