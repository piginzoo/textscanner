import logging
import time
import os
from logging import handlers
import datetime
import tensorflow as tf

import conf
from utils.util import logger


def init(level=logging.DEBUG, when="D", backup=7,
         _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d %(message)s"):
    train_start_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    filename = 'logs/textscanner-' + train_start_time + '.log'
    _dir = os.path.dirname(filename)
    if not os.path.isdir(_dir): os.makedirs(_dir)

    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter(_format)
        logger.setLevel(level)

        handler = handlers.TimedRotatingFileHandler(filename, when=when, backupCount=backup, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def call_debug(layer, *input):
    if not conf.DEBUG:
        return layer(*input)

    layer_name = "Unknown"
    if hasattr(layer, "__name__"):
        layer_name = layer.__name__
    if hasattr(layer, "name"):
        layer_name = layer.name

    input_shape = "Unknown"
    if type(input[0]) == list:
        input_shape = str([str(i.shape) for i in input[0]])
    if hasattr(input[0], "shape"):
        input_shape = str(input[0].shape)

    assert callable(layer), "layer[" + layer_name + "] is callable"
    output = layer(*input)

    if type(output) == list or type(output) == tuple:
        output_shape = str([str(o.shape) for o in output])
    else:
        output_shape = str(output.shape)

    logger.debug("Layer: {:25s}    {:30s} => {:30s}".format(layer_name, input_shape, output_shape))
    return output
