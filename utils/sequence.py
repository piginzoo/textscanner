from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.label.label_maker import LabelGenerater
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from utils.label.label import ImageLabel
from utils.label import label_utils
from utils.data_util import ImageLabelLoader
import time, cv2, os
import logging, math
import numpy as np

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
        self.target_image_shape = (conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH)
        self.label_generator = LabelGenerater(conf.MAX_SEQUENCE, self.target_image_shape, charsets)
        self.image_loader = ImageLabelLoader(self.target_image_shape, self.charsets, "plaintxt", conf.MAX_SEQUENCE)

        def __len__(self):
            return int(math.ceil(len(self.data_list) / self.batch_size))

        def __getitem__(self, idx):
            batch_data_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]
            images, labels = self.image_loader.load_image_label(batch_data_list)
            return images, labels

        def on_epoch_end(self):
            np.random.shuffle(self.data_list)
            duration = time.time() - self.start_time
            self.start_time = time.time()
            logger.debug("[%s] Epoch done, elapsed time[%d]s，re-shuffle", self.name, duration)

        def initialize(self, args):
            logger.info("[%s]begin to load image/labels", self.name)
            start_time = time.time()
            self.data_list = label_utils.load_labels(self.label_dir)
            if len(self.data_list) == 0:
                msg = f"[{self.name}] 图像和标签加载失败[目录：{self.label_dir}]，0条！"
                raise ValueError(msg)

            logger.info("[%s]loaded [%d] labels,elapsed time [%d]s", self.name, len(self.data_list),
                        (time.time() - start_time))
