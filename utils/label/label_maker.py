from utils.label import label_utils
import scipy.ndimage.filters as fi
from utils import image_utils
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)


class LabelGenerater():
    """
    The class is used to generate the GT labels.
    In loss function, we need 3 GT: Q,H,G
    Refer to : http://www.piginzoo.com/machine-learning/2020/04/14/ocr-fa-textscanner#%E5%85%B3%E4%BA%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0
    - Order map GT : H
    - Localization map GT : Q
    - Character Segmentation : G
    """
    shrink = 1  # shrink ratio for one character wrapped polygon
    ζ = 0.5  # threshold for normalization
    δ = 5  # variation for Gaussian distribution

    def __init__(self, max_sequence, target_image_shape, charset):
        self.max_sequence = max_sequence
        self.target_image_shape = target_image_shape  # [H,W]: [64,256]
        self.target_width = target_image_shape[1]
        self.target_height = target_image_shape[0]
        self.charset = charset

    # # adjust all polygens' co-ordinations
    # def _adjust_by_size(self, boxes, original_shape):
    #     assert len(boxes.shape) == 2 or len(boxes.shape) == 3
    #
    #     ratio_x = original_shape[1] / self.target_width
    #     ratio_y = original_shape[0] / self.target_height
    #
    #     if len(boxes.shape) == 3:
    #         boxes[:, :, 0] = (boxes[:, :, 0] / ratio_x).clip(0, self.target_width)
    #         boxes[:, :, 1] = (boxes[:, :, 1] / ratio_y).clip(0, self.target_heigth)
    #     else:
    #         boxes[:, 0] = (boxes[:, 0] / ratio_x).clip(0, self.target_width)
    #         boxes[:, 1] = (boxes[:, 1] / ratio_y).clip(0, self.target_heigth)
    #
    #     boxes = (boxes + .5).astype(np.int32)
    #     return boxes

    # data is ImageLabel{image,[Label]}
    def process(self, image_labels):

        # adjust the coordination
        shape = image_labels.image.shape[:2]  # h,w
        boxes = image_labels.bboxes  # [N,4,2] N: words number
        label = image_labels.label

        # # find the one bbox boundary
        # xmins = boxes[:, :, 0].min(axis=1)
        # xmaxs = np.maximum(boxes[:, :, 0].max(axis=1), xmins + 1)
        # ymins = boxes[:, :, 1].min(axis=1)
        # ymaxs = np.maximum(boxes[:, :, 1].max(axis=1), ymins + 1)

        character_segment = self.render_character_segemention(image_labels)
        localization_map = np.zeros(self.target_image_shape, dtype=np.float32)
        order_segments = np.zeros((*self.target_image_shape, self.max_sequence), dtype=np.float32)
        order_maps = np.zeros((*self.target_image_shape, self.max_sequence), dtype=np.float32)

        assert boxes.shape[0] <= self.max_sequence, \
            f"the train/validate label text length[{len(image_labels.labels)}] must be less than pre-defined max sequence length[{self.max_sequence}]"

        # process each character
        for i in range(boxes.shape[0]):
            # Y_hat_k is the normalized_gaussian map, comply with the name in the paper
            Y_hat_k = self.generate_Y_hat_k_by_gaussian_normalize(self.target_image_shape,
                                                                  boxes[i])  # xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            if Y_hat_k is None:
                logger.warning("Y_%d generator failed,the char[%s] of [%s]", i, label[i], label)
                Y_hat_k = np.zeros((self.target_image_shape))

            self.render_order_segment(order_segments[:, :, i], Y_hat_k, threshold=self.ζ)
            localization_map = self.render_localization_map(localization_map, Y_hat_k)
            order_maps = order_segments * localization_map[:, :, np.newaxis]

        return character_segment, order_segments, order_maps, localization_map

    # 围绕中心点做一个高斯分布，但是由于每个点的概率值过小，所以要做一个归一化,使得每个点的值归一化到[0,1]之间
    # Make a gaussian distribution with the center, and do normalization
    # def gaussian_normalize(self, shape, xmin, xmax, ymin, ymax)：
    # @return a "image" with shape[H,W], which is filled by a gaussian distribution
    def generate_Y_hat_k_by_gaussian_normalize(self, shape, one_word_bboxes):  # one_word_bboxes[4,2]
        # logger.debug("The word bbox : %r , image shape is : %r", one_word_bboxes, shape)

        # find the one bbox boundary
        xmin = one_word_bboxes[:, 0].min()
        xmax = one_word_bboxes[:, 0].max()
        ymin = one_word_bboxes[:, 1].min()
        ymax = one_word_bboxes[:, 1].max()

        out = np.zeros(shape)
        h, w = shape[:2]
        # find the "Center" of polygon
        y = (ymax + ymin + 1) // 2
        x = (xmax + xmin + 1) // 2
        if x >= w or y >= h:
            logger.warning("标注超出图像范围，生成高斯样本失败：(xmin:%f, xmax:%f, ymin:%f, ymax:%f,w:%f,x:%f,h:%f,y:%f)", xmin, xmax,
                           ymin, ymax, w, x, h, y)
            return None

        # prepare the gaussian distribution,refer to paper <<Label Generation>>
        out[y, x] = 1.

        fi.gaussian_filter(out, (self.δ, self.δ), output=out, mode='mirror')

        # logger.debug("Max gaussian value is :%f", out.max()) # it is 0.006367
        if out is None: return None

        return out

    def render_order_segment(self, order_maps, Y_k, threshold):
        Z_hat_k = Y_k / Y_k.max()
        Z_hat_k[Z_hat_k < threshold] = 0
        # Z_hat_k[Z_hat_k >= threshold] = 1
        order_maps[:] = Z_hat_k

    # fill the shrunk zone with the value of character ID
    def render_character_segemention(self, image_labels):

        character_segment = np.zeros(self.target_image_shape, dtype=np.int32)

        for one_word_label in image_labels.labels:
            label = one_word_label.label
            char_id = label_utils.str2id(label, self.charset)

            # shrink one word bboxes to avoid overlap
            shrinked_poly = image_utils.shrink_poly(one_word_label.bbox, self.shrink)

            word_fill = np.zeros(self.target_image_shape, np.uint32)

            word_fill.fill(char_id)

            mask = np.zeros(self.target_image_shape, np.uint8)
            cv2.fillPoly(mask, [shrinked_poly], 1)  # set those words' bbox area value to 1
            character_segment = np.maximum(mask * word_fill, character_segment) # merge two, only by maximum, not add
            character_segment.astype(np.int32)

        return character_segment

    # merge all Y_k with Max value
    def render_localization_map(self, localization_map, Y_k):
        return np.maximum(localization_map, Y_k)
