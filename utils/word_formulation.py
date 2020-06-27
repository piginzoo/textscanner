from utils.label import label_utils
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def word_formulate(G, H):
    """
    Calculate each word's possibility.
    :param  G
        Character Segmentation, [B,H,W,C:4100]
    :param  H
        Order Map, [B,H,W,S:30]
    :return
        list, contain char ids, and if 0 appear, the process break, 0 means background
    """

    ids = []
    for i in range(H.shape[-1]):
        # G[H,W,C:4100] * H_k[H,W,1]
        # G是每个像素字符的概率(1/4100)
        # H_k是第k个字符对应的正态分布
        # (G*H_k) ===> [H,W,4100]
        # sum = \sum(G*H_k) ===> [4100]
        _H_k = H[:,:,:,i]
        _H_k = H_k[:, :, np.newaxis]  # [H,W] => [H,W,1]
        GH_k = (G * _H_k)
        sum = np.sum(GH_k, axis=(0, 1))
        id = sum.argmax()
        if id == 0: break
        ids.append(id)

    return ids


def process(character_segment_G, charset, image_label, order_maps_H):
    """
    This is just for debug, called by test cases
    :param character_segment_G:
    :param charset:
    :param image_label:
    :param order_maps_H:
    :return:
    """
    start = time.time()
    G = np.eye(len(charset))[character_segment_G]  # eye是对角阵生成函数，通过他，完成categorical one hot化
    print("G:",G.shape)
    print("H:",order_maps_H.shape)
    ids = word_formulate(G, order_maps_H)
    pred = label_utils.id2str(ids, charset)
    if image_label.label != pred:
        print("Predict:[%s]" % pred)
        print("Label  :[%s]" % image_label.label)

    t = time.time() - start
    print("Word Formulation time: %f" % t)
    return t
