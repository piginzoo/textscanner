from tensorflow.keras.callbacks import Callback
from utils import word_formulation
import tensorflow as tf
from utils import util
import numpy as np
import logging
import conf
import time

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """
    Since OOM issue, I move the accuracy metrics out of the tensor graph,
    It will use CPU instead of GPU,
    """

    def __init__(self, image_loader, steps, batch, tb_dir):
        self.name = "Validation"
        self.best_accuracy = 0
        self.image_loader = image_loader
        self.batch = batch
        self.steps = steps
        self.file_writer = tf.summary.create_file_writer(tb_dir + "/metrics")



    def on_epoch_end(self, epoch, logs=None):

        start = time.time()

        self.image_loader.shuffle()
        all_data = self.image_loader.data_list[:self.batch * self.steps]

        logger.info("[Validation] start: epoch: #%d, validation size: %d", epoch, len(all_data))

        all_pred_ids = []
        all_label_ids = []
        for i in range(self.steps):
            try:
                print("============================")
                print(str(i * self.batch),"=>",str((i + 1) * self.batch))
                data = all_data[i * self.batch: (i + 1) * self.batch]
                images, labels = self.image_loader.load_image_label(data)


                print(images.shape)
                pred = self.model(images)  # return [character_segment, order_map, localization_map]

                pred_character_segments = pred['character_segmentation']  # np.argmax(pred['character_segmentation'], axis=-1)
                pred_order_maps = pred['order_map']
                # logger.debug("G:",pred_character_segments.shape)
                # logger.debug("H:",pred_order_maps.shape)
                pred_ids = word_formulation.word_formulate(G=pred_character_segments, H=pred_order_maps)
                label_ids = labels['label_ids']
                all_pred_ids.append(pred_ids)
                all_label_ids.append(label_ids)
            except Exception as e:
                logger.exception("Error during validating")

        all_pred_ids = np.stack(all_pred_ids)
        all_label_ids = np.stack(all_label_ids)

        acc = self._accuracy(all_pred_ids, all_label_ids)

        end = time.time()
        logger.info("[Validation] end: epoch: #%d, validation size: %d", epoch, len(all_data))
        logger.info("[Validation] Accuracy: %f , time elapse: %s seconds", acc, (end - start))
        tf.summary.scalar('Epoch Accuracy', data=acc, step=epoch)

        if (acc > self.best_accuracy):
            logger.info("The current accuracy[%f] is better than ever[%f]", acc, self.best_accuracy)
            self.best_accuracy = acc
            self._save_model(epoch, acc)



        return

    def _save_model(self, epoch, acc):
        timestamp = util.timestamp_s()
        model_path = f"{conf.DIR_MODEL}/model-{timestamp}-epoch{epoch}-acc{acc}.pb"
        self.model.save(model_path, save_format='tf')
        logger.info("The current model was saved : %s", model_path)

    def _accuracy(self, pred_ids, label_ids):
        batch_equals = pred_ids==label_ids
        batch_equals = np.all(batch_equals,axis=1)
        acc = np.mean(batch_equals)
        return acc

if __name__=="__main__":
    mc = MetricsCallback(None,None,None)

    pred_ids = np.random.randint(0,100,(10,30))
    label_ids = np.random.randint(0, 100, (10, 30))
    acc = mc._accuracy(pred_ids, label_ids)
    print("acc:",acc)

    pred_ids = np.full((10,30),23)
    label_ids = np.full((10,30),23)
    acc = mc._accuracy(pred_ids, label_ids)
    print("acc:",acc)

    pred_ids = np.full((10,30),23)
    label_ids = np.full((10,30),0)
    label_ids[:5,:] = 23
    acc = mc._accuracy(pred_ids, label_ids)
    print("acc:",acc)
