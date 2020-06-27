from utils.label.label_maker import LabelGenerater
from utils.label.image_label_loader import ImageLabelLoader
from tensorflow.keras.callbacks import Callback
from PIL import ImageFont
import numpy as np
import logging
import conf
import time
from utils.label import label_utils
from utils import word_formulation
from utils import util

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):

    def __init__(self, image_loader):
        self.name = "Validation"
        self.best_accuracy = 0
        self.image_loader = image_loader

    def on_epoch_end(self, epoch, logs=None):

        self.image_loader.shuffle()
        data = self.image_loader.data_list[:3]
        images, labels = self.image_loader.load_image_label(data)

        pred = self.model(images)  # return [character_segment, order_map, localization_map]

        pred_character_segments = np.argmax(pred['character_segmentation'], axis=-1)
        pred_order_maps = pred['order_map']
        pred_ids = word_formulation.word_formulate(G=pred_character_segments, H=pred_order_maps)
        label_ids = labels['label_ids']

        acc = self._accuracy(pred_ids, label_ids)
        if (acc > self.best_accuracy):
            logger.info("The current accuracy[%f] is better than ever[%f]", acc, self.best_accuracy)
            self.best_accuracy = acc
            self._save_model(epoch,acc)

        return

    def _save_model(self, epoch, acc):
        timestamp = util.timestamp_s()
        model_path = f"${conf.DIR_MODEL}/model-${timestamp}-epoch${epoch}-acc${acc}.hdf5"
        self.model.save(model_path)
        logger.info("The current model was saved : %s", model_path)

    def _accuracy(self, pred_ids, label_ids):
        correct = 0
        for i in range(len(pred_ids)):
            pred_id = pred_ids[i]
            label_id = label_ids[i]

            if len(pred_id) != len(label_id): continue

            b_same = True
            for j in range(len(pred_id)):
                if pred_id[j] != label_id[j]:  # any id different, fail
                    b_same = False
                    break
            if b_same: correct += 1
        return correct / len(pred_ids)

