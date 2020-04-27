from network.model import TextScannerModel
from utils.sequence import SequenceData
from utils import util, logger as log,label_utils
import os
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from tensorflow.python.keras.models import load_model
from main import conf
import logging

logger = logging.getLogger("Train")


def train(args):

    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)

    model = TextScannerModel()

    train_sequence = SequenceData(name="训练",
                                  label_file=args.train_label_file,
                                  charset_file=conf.CHARSET,
                                  conf=conf,
                                  args=args,
                                  batch_size=args.batch)
    valid_sequence = SequenceData(name="验证",
                                  label_file=args.validate_label_file,
                                  charset_file=conf.CHARSET,
                                  conf=conf,
                                  args=args,
                                  batch_size=args.validation_batch)

    timestamp = util.timestamp_s()
    tb_log_name = os.path.join(conf.DIR_TBOARD,timestamp)
    checkpoint_path = conf.DIR_MODEL + "/model-" + timestamp + "-epoch{epoch:03d}-acc{words_accuracy:.4f}-val{val_words_accuracy:.4f}.hdf5"

    # 如果checkpoint文件存在，就加载之
    if args.retrain:
        logger.info("重新开始训练....")
    else:
        logger.info("基于之前的checkpoint训练...")
        _checkpoint_path = util.get_checkpoint(conf.DIR_CHECKPOINT)
        if _checkpoint_path is not None:
            model = load_model(_checkpoint_path)
            logger.info("加载checkpoint模型[%s]", _checkpoint_path)
        else:
            logger.warning("找不到任何checkpoint，重新开始训练")

    logger.info("Begin train开始训练：")

    tboard = TensorBoard(log_dir=tb_log_name,histogram_freq=1,batch_size=2,write_grads=True)
    #early_stop = EarlyStopping(monitor='words_accuracy', patience=args.early_stop, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, mode='max')

    model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=args.steps_per_epoch,#其实应该是用len(train_sequence)，但是这样太慢了，所以，我规定用一个比较小的数，比如1000
        epochs=args.epochs,
        workers=args.workers,   # 同时启动多少个进程加载
        callbacks=[tboard,checkpoint],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=args.validation_steps)

    logger.info("Train end训练结束!")

    model_path = conf.DIR_MODEL+"/ocr-attention-{}.hdf5".format(util.timestamp_s())
    model.save(model_path)
    logger.info("Save model保存训练后的模型到：%s", model_path)

if __name__ == "__main__":
    log.init()
    args = conf.init_args()
    train(args)