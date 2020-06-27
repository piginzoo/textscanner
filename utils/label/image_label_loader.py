from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.label.label_maker import LabelGenerater
from tensorflow.keras.utils import to_categorical
from utils.label.label import ImageLabel
from utils.label import label_utils
import numpy as np
import cv2, os
import logging
import conf
import time

logger = logging.getLogger(__name__)


class ImageLabelLoader:

    def __init__(self, name, label_dir, target_image_shape, charsets, label_format, max_squence):
        self.name = name
        self.charsets = charsets
        self.label_format = label_format
        self.target_image_shape = target_image_shape
        self.max_sequence = max_squence
        self.label_generator = LabelGenerater(max_squence, self.target_image_shape, charsets)
        self.initialize(label_dir)


    def initialize(self, label_dir):
        logger.info("[%s]begin to load image/labels", self.name)
        start_time = time.time()
        self.data_list = label_utils.load_labels(label_dir)
        if len(self.data_list) == 0:
            msg = f"[{self.name}] 图像和标签加载失败[目录：{label_dir}]，0条！"
            raise ValueError(msg)
        logger.info("[%s]loaded [%d] labels,elapsed time [%d]s", self.name, len(self.data_list),
                    (time.time() - start_time))

    def shuffle(self):
        np.random.shuffle(self.data_list)

    def load_image_label(self, batch_data_list):
        images = []
        batch_cs = []  # Character Segment
        batch_os = []  # Order Segment
        batch_om = []  # Order Map
        batch_lm = []  # Localization Map
        label_ids = [] # label ids[[21,24,256,2,121],[...],...]
        for image_path, label_path in batch_data_list:

            if not os.path.exists(image_path):
                logger.warning("Image [%s] does not exist", image_path)
                continue

            character_segment, localization_map, order_sgementation, order_maps, image, label_id = \
                self.load_one_image_label(image_path, label_path)

            images.append(image)
            batch_cs.append(character_segment)
            batch_os.append(order_sgementation)
            batch_lm.append(localization_map)
            batch_om.append(order_maps)
            label_ids.append(label_id)

        images = np.array(images, np.float32)
        label_ids = pad_sequences(label_ids, maxlen=conf.MAX_SEQUENCE, padding="post", value=0)
        label_ids = np.array(label_ids)
        batch_cs = np.array(batch_cs)
        batch_om = np.array(batch_om)
        batch_os = np.array(batch_os)
        batch_lm = np.array(batch_lm)

        # logger.debug("Loaded images:  %r", images.shape)
        # logger.debug("Loaded batch_cs:%r", batch_cs.shape)
        # logger.debug("Loaded batch_om:%r", batch_om.shape)
        # logger.debug("Loaded batch_lm:%r", batch_lm.shape)
        # logger.debug("[%s] loaded %d data", name, len(images))

        return images, {'character_segmentation': batch_cs,
                        'order_map': batch_om,
                        'localization_map': batch_lm,
                        'label_ids': label_ids}

    def load_one_image_label(self, image_path, label_path):

        label_file = open(label_path, encoding="utf-8")
        data = label_file.readlines()
        label_file.close()
        if conf.DEBUG: logger.debug("Loaded label file [%s] %d lines", label_path, len(data))
        target_size = (self.target_image_shape[1], self.target_image_shape[0])
        # inside it, the bboxes size will be adjust
        il = ImageLabel(cv2.imread(image_path), data, self.label_format, target_size=target_size)

        # text label
        label = il.label
        label_ids = label_utils.strs2id(label, self.charsets)

        # character_segment, order_maps, localization_map = label_generator.process(il)
        character_segment, order_sgementation, order_maps, localization_map = self.label_generator.process(il)
        character_segment = to_categorical(character_segment,
                                           num_classes=len(self.charsets) + 1)  # <--- size becoming big!!!
        return character_segment, localization_map, order_sgementation, order_maps, il.image, label_ids
