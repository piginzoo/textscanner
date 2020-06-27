from utils.label.image_label_loader import ImageLabelLoader
from tensorflow.keras.utils import Sequence
import logging, math
import conf
import time

logger = logging.getLogger("SequenceData")


class SequenceData(Sequence):

    def __init__(self, image_loader, batch_size=32):
        self.batch_size = batch_size
        self.image_loader = image_loader
        self.start_time = time.time()

    def __len__(self):
        return int(math.ceil(len(self.image_loader.data_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_data_list = self.image_loader.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        if conf.DEBUG: logger.debug("Load batch data,idx=%d", idx)
        images, labels = self.image_loader.load_image_label(batch_data_list)
        # must pop 'label_ids' key, or it will cause OP nest.flatten disorder
        labels.pop("label_ids")
        return images, labels

    def on_epoch_end(self):
        self.image_loader.shuffle()