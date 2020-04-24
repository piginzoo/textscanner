from network import model as _model
from network.attention import AttentionLayer
from utils.sequence import SequenceData
from utils import util, logger as log,label_utils
import os
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from tensorflow.python.keras.models import load_model
from main import conf
import logging
from utils.visualise_attention import  TBoardVisual

logger = logging.getLogger("Train")

def train(args):
    # TF调试代码 for tf debugging：
    # from tensorflow.python import debug as tf_debug
    # from tensorflow.python.keras import backend as K
    # sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)

    model, _, _ = _model.model(conf, args)

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
            model = load_model(_checkpoint_path,
                custom_objects={
                    'words_accuracy': _model.words_accuracy,
                    'AttentionLayer':AttentionLayer})
            logger.info("加载checkpoint模型[%s]", _checkpoint_path)
        else:
            logger.warning("找不到任何checkpoint，重新开始训练")

    logger.info("Begin train开始训练：")

    attention_visible = TBoardVisual('Attetnon Visibility',tb_log_name,charset,args)
    tboard = TensorBoard(log_dir=tb_log_name,histogram_freq=1,batch_size=2,write_grads=True)
    early_stop = EarlyStopping(monitor='words_accuracy', patience=args.early_stop, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='words_accuracy', verbose=1, mode='max')

    model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=args.steps_per_epoch,#其实应该是用len(train_sequence)，但是这样太慢了，所以，我规定用一个比较小的数，比如1000
        epochs=args.epochs,
        workers=args.workers,   # 同时启动多少个进程加载
        callbacks=[tboard,checkpoint,early_stop,attention_visible],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=args.validation_steps)
    # [validation_steps](https://keras.io/zh/models/model/)：
    # 对于 Sequence，它是可选的：如果未指定，将使用 len(generator) 作为步数。

    logger.info("Train end训练结束!")

    model_path = conf.DIR_MODEL+"/ocr-attention-{}.hdf5".format(util.timestamp_s())
    model.save(model_path)
    logger.info("Save model保存训练后的模型到：%s", model_path)


if __name__ == "__main__":
    log.init()
    args = conf.init_args()
    #with K.get_session(): # 防止bug：https://stackoverflow.com/questions/40560795/tensorflow-attributeerror-nonetype-object-has-no-attribute-tf-deletestatus
    #     with tf.device("/device:GPU:0"):

    train(args)