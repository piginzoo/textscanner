from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from utils.label.label import ImageLabel
from utils.sequence import SequenceData
from utils.label import label_utils
import numpy as np
import cv2, os
import logging

logger = logging.getLogger("SequenceData")


class ValidationSequenceData(SequenceData):

    def load_image_label(self, batch_data_list):

        images = []
        labels = []
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

            images.append(il.image)
            label = il.label
            label_ids = label_utils.strs2id(label, self.charsets)
            labels.append(label_ids)

        # import pdb; pdb.set_trace()
        labels = pad_sequences(labels, maxlen=self.conf.MAX_SEQUENCE, padding="post", value=0)
        labels = to_categorical(labels, num_classes=len(self.charsets))
        images = np.array(images, np.float32)
        return images, np.array(labels)