#!/usr/bin/env python
import time,os,datetime,logging

logger = logging.getLogger("Util")


def call_debug(layer, *input):

    layer_name = "Unknown"
    if hasattr(layer, "__name__"):
        layer_name = layer.__name__
    if hasattr(layer, "name"):
        layer_name = layer.name

    input_shape = "Unknown"
    if type(input[0]) == list:
        input_shape = str([str(i.shape) for i in input[0]])
    if hasattr(input[0],"shape"):
        input_shape = str(input[0].shape)

    assert callable(layer), "layer[" + layer_name + "] is callable"
    output = layer(*input)

    if type(output) == list or type(output) == tuple:
        output_shape = str([str(o.shape) for o in output])
    else:
        output_shape = str(output.shape)

    print("Layer: {:25s}    {:30s} => {:30s}".format(layer_name, input_shape, output_shape))
    return output


def timestamp_s():
    s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    return s


def get_checkpoint(dir):
    if not os.path.exists(dir):
        logger.info("找不到最新的checkpoint文件")
        return None

    list = os.listdir(dir)
    if len(list)==0:
        logger.info("找不到最新的checkpoint文件")
        return None

    list.sort(key=lambda fn:os.path.getmtime( os.path.join(dir,fn) ))

    latest_model_name = os.path.join(dir,list[-1])

    logger.debug("在目录%s中找到最新的模型文件：%s",dir,latest_model_name)

    return latest_model_name
