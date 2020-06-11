from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.label.label_maker import LabelGenerater
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from utils.label.label import ImageLabel
from utils.label import label_utils
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

    def __len__(self):
        return int(math.ceil(len(self.data_list) / self.batch_size))

    def load_image_label(self, batch_data_list):

        images = []
        batch_cs = []   # Character Segment
        batch_os = []   # Order Segment
        # batch_om = [] # Order Map
        batch_lm = []   # Localization Map
        label_text = [] # label text
        for image_path, label_path in batch_data_list:

            if not os.path.exists(image_path):
                logger.warning("Image [%s] does not exist", image_path)
                continue

            label_file = open(label_path, encoding="utf-8")
            data = label_file.readlines()
            label_file.close()
            logger.debug("Loaded label file [%s] %d lines", label_path, len(data))
            target_size = (self.target_image_shape[1], self.target_image_shape[0])
            il = ImageLabel(cv2.imread(image_path), data, self.conf.LABLE_FORMAT,
                            target_size=target_size)  # inside it, the bboxes size will be adjust
            logger.debug("Loaded label generates training labels")

            images.append(il.image)

            # text label
            label = il.label
            label_ids = label_utils.strs2id(label, self.charsets)
            label_text.append(label_ids)

            # character_segment, order_maps, localization_map = self.label_generator.process(il)
            character_segment, order_sgementation, localization_map = self.label_generator.process(il)
            character_segment = to_categorical(character_segment, num_classes=len(self.charsets) + 1)

            batch_cs.append(character_segment)
            # batch_om.append(order_maps)
            batch_os.append(order_sgementation)
            batch_lm.append(localization_map)

        images = np.array(images, np.float32)
        batch_cs = np.array(batch_cs)
        # batch_om = np.array(batch_om)
        batch_os = np.array(batch_os)
        batch_lm = np.array(batch_lm)

        # text one hot array
        labels = pad_sequences(label_text, maxlen=self.conf.MAX_SEQUENCE, padding="post", value=0)
        labels = to_categorical(labels, num_classes=len(self.charsets))

        # logger.debug("Loaded images:  %r", images.shape)
        # logger.debug("Loaded batch_cs:%r", batch_cs.shape)
        # logger.debug("Loaded batch_om:%r", batch_om.shape)
        # logger.debug("Loaded batch_lm:%r", batch_lm.shape)
        # logger.debug("[%s] loaded %d data", self.name, len(images))

        return images, [batch_cs, batch_os, batch_lm,labels]

    def __getitem__(self, idx):
        batch_data_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        images, labels = self.load_image_label(batch_data_list)
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
