from utils.callbacks.visualise_callback import VisualCallback
from utils.callbacks.metrics_callback import MetricsCallback
from utils.label.image_label_loader import ImageLabelLoader
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from network.model import TextScannerModel
from utils.sequence import SequenceData
from utils.label import label_utils
from utils import logger as log
import tensorflow as tf
from utils import util
import logging
import conf
import os

logger = logging.getLogger(__name__)


def train(args):
    # limit the GPU memory over occupy
    limit_gpu_memory_over_occupy()

    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)

    model = TextScannerModel(conf, charset)

    target_image_shape = (conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH)
    train_image_loader = ImageLabelLoader("Train", args.train_label_dir, target_image_shape, charset, "plaintext", conf.MAX_SEQUENCE)
    train_sequence = SequenceData(image_loader=train_image_loader, batch_size=args.batch)

    timestamp = util.timestamp_s()
    tb_log_name = os.path.join(conf.DIR_TBOARD, timestamp)
    checkpoint_path = conf.DIR_MODEL + "/model-" + timestamp + "-epoch{epoch:03d}.hdf5"

    # 如果checkpoint文件存在，就加载之
    if args.retrain:
        logger.info("Train from beginning ...")
    else:
        logger.info("基于之前的checkpoint训练...")
        _checkpoint_path = util.get_checkpoint(conf.DIR_CHECKPOINT)
        if _checkpoint_path is not None:
            model = load_model(_checkpoint_path)
            logger.info("加载checkpoint模型[%s]", _checkpoint_path)
        else:
            logger.warning("找不到任何checkpoint，重新开始训练")

    logger.info("Train begin：")

    # update tboard scalar per 100 batches, profile no. 2 batch of each epoch
    validate_image_loader = ImageLabelLoader("Validation", args.validate_label_dir, target_image_shape, charset, "plaintext", conf.MAX_SEQUENCE)
    tboard = TensorBoard(log_dir=tb_log_name, update_freq=conf.UPDATE_FREQ, profile_batch=0, histogram_freq=0)
    early_stop = EarlyStopping(patience=args.early_stop, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, mode='max')
    visibility_debug = VisualCallback('Attetnon Visibility', tb_log_name, validate_image_loader, args.debug_step)
    metrics = MetricsCallback(image_loader=validate_image_loader,
                              steps=args.validation_batch * args.validation_steps,
                              batch=args.validation_batch)

    model.comile_model()

    model.fit(
        x=train_sequence,
        steps_per_epoch=args.steps_per_epoch,  # 其实应该是用len(train_sequence)，但是这样太慢了，所以，我规定用一个比较小的数，比如1000
        epochs=args.epochs,
        workers=args.workers,  # 同时启动多少个进程加载
        callbacks=[tboard, checkpoint, early_stop, visibility_debug, metrics],
        use_multiprocessing=True,
        verbose=2)

    logger.info("Train end!")

    model_path = conf.DIR_MODEL + "/textscanner-{}.hdf5".format(util.timestamp_s())
    # model.save(model_path)
    model.save_weights(model_path)
    logger.info("Save model saved to ：%s", model_path)


def limit_gpu_memory_over_occupy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    log.init()
    args = conf.init_args()
    train(args)
