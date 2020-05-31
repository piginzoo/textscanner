#!/usr/bin/env python
import numpy as np
import logging
import time
import conf
import cv2
import os

logger = logging.getLogger(__name__)


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


def timestamp_s():
    s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    return s


# bboxes[4,2]
def resize_bboxes(bboxes, original_size, target_size):
    logger.debug("Adjust bboxes from %r => %r", original_size, target_size)
    logger.debug("Original:%r",bboxes)

    w, h = original_size
    ratio_x = target_size[0] / w
    ratio_y = target_size[1] / h

    logger.debug("ratio_x:%r",ratio_x)
    logger.debug("ratio_y:%r", ratio_y)

    # bboxes: [4,2]
    bboxes = np.array(bboxes)
    bboxes[:, 0] = bboxes[:, 0] * ratio_x
    bboxes[:, 1] = bboxes[:, 1] * ratio_y

    # if the coordinator is minus, set to 0
    minus_indices = bboxes < 0
    bboxes[minus_indices] = 0

    # adjust out boundary for width to max width
    out_bound_indices = bboxes[:, 0] > target_size[0]
    bboxes[out_bound_indices, 0] = target_size[0]

    # adjust out boundary for height to max height
    out_bound_indices = bboxes[:, 1] > target_size[1]
    bboxes[out_bound_indices, 1] = target_size[1]

    logger.debug("Resized:%r",bboxes)

    return bboxes.tolist()


def get_checkpoint(dir):
    if not os.path.exists(dir):
        logger.info("找不到最新的checkpoint文件")
        return None

    list = os.listdir(dir)
    if len(list) == 0:
        logger.info("找不到最新的checkpoint文件")
        return None

    list.sort(key=lambda fn: os.path.getmtime(os.path.join(dir, fn)))

    latest_model_name = os.path.join(dir, list[-1])

    logger.debug("在目录%s中找到最新的模型文件：%s", dir, latest_model_name)

    return latest_model_name
