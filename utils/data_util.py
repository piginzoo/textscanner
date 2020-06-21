from tensorflow.keras.utils import to_categorical
from utils.label.label_maker import LabelGenerater
from utils.label.label import ImageLabel
from utils.label import label_utils
import numpy as np
import cv2, os
import logging

logger = logging.getLogger(__name__)


class ImageLabelLoader:

    def __init__(self, target_image_shape, charset, label_format, max_squence):
        self.charsets = charset
        self.label_format = label_format
        self.target_image_shape = target_image_shape
        self.max_sequence = max_squence
        self.label_generator = LabelGenerater(max_squence, self.target_image_shape, charset)

    def load_image_label(self, batch_data_list):
        images = []
        batch_cs = []  # Character Segment
        batch_os = []  # Order Segment
        # batch_om = [] # Order Map
        batch_lm = []  # Localization Map
        label_text = []  # label text
        for image_path, label_path in batch_data_list:

            if not os.path.exists(image_path):
                logger.warning("Image [%s] does not exist", image_path)
                continue

            character_segment, localization_map, order_sgementation, image, label_text = \
                self.load_one_image_label(image_path, label_path, label_text)

            batch_cs.append(character_segment)
            batch_os.append(order_sgementation)
            batch_lm.append(localization_map)
            images.append(image)
            # batch_om.append(order_maps)

        images = np.array(images, np.float32)
        label_text.append(label_text)
        batch_cs = np.array(batch_cs)
        # batch_om = np.array(batch_om)
        batch_os = np.array(batch_os)
        batch_lm = np.array(batch_lm)

        # text one hot array
        # labels = pad_sequences(label_text, maxlen=conf.MAX_SEQUENCE, padding="post", value=0)
        # labels = to_categorical(labels, num_classes=len(charsets))

        # logger.debug("Loaded images:  %r", images.shape)
        # logger.debug("Loaded batch_cs:%r", batch_cs.shape)
        # logger.debug("Loaded batch_om:%r", batch_om.shape)
        # logger.debug("Loaded batch_lm:%r", batch_lm.shape)
        # logger.debug("[%s] loaded %d data", name, len(images))

        return images, [batch_cs, batch_os, batch_lm]  # ,labels]

    def load_one_image_label(self, data):
        # print(data)
        #import pdb; pdb.set_trace()
        image_path, label_path = data[0],data[1]
        label_file = open(label_path, encoding="utf-8")
        data = label_file.readlines()
        label_file.close()
        logger.debug("Loaded label file [%s] %d lines", label_path, len(data))
        target_size = (self.target_image_shape[1], self.target_image_shape[0])
        # inside it, the bboxes size will be adjust
        il = ImageLabel(cv2.imread(image_path), data, self.label_format, target_size=target_size)
        logger.debug("Loaded label generates training labels")

        # text label
        label = il.label
        label_ids = label_utils.strs2id(label, self.charset)

        # character_segment, order_maps, localization_map = label_generator.process(il)
        character_segment, order_sgementation, localization_map = self.label_generator.process(il)
        character_segment = to_categorical(character_segment,
                                           num_classes=len(self.charset) + 1)  # <--- size becoming big!!!
        return character_segment, localization_map, order_sgementation, il.image, label_ids
