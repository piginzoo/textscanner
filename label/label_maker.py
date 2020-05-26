import cv2
import numpy as np
import scipy.ndimage.filters as fi
from utils import image_utils
from utils import label_utils


class LabelGenerater():
    """
    The class is used to generate the GT labels.
    In loss function, we need 3 GT: Q,H,G
    Refer to : http://www.piginzoo.com/machine-learning/2020/04/14/ocr-fa-textscanner#%E5%85%B3%E4%BA%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0
    - Order map GT : H
    - Localization map GT : Q
    - Character Segmentation : G
    """
    shrink = 0.6    # shrink ratio for one character wrapped polygon
    ζ = 0.5         # threshold for normalization
    δ = 5           # variation for Gaussian distribution

    def __init__(self,max_sequence,target_image_shape,charset):
        self.max_sequence = max_sequence
        self.target_image_shape = target_image_shape # default:[64,256]
        self.charset = charset

    def _adjust_by_size(self,boxes,original_shape):
        assert len(boxes.shape)==2 or len(boxes.shape)==3

        ratio_x = original_shape[1] / self.target_image_shape[1]
        ratio_y = original_shape[0] / self.target_image_shape[0]

        if len(boxes.shape)==3:
            boxes[:, :, 0] = (boxes[:, :, 0] / ratio_x).clip(0, self.target_image_shape[1])
            boxes[:, :, 1] = (boxes[:, :, 1] / ratio_y).clip(0, self.target_image_shape[0])
        else:
            boxes[   :, 0] = (boxes[   :, 0] / ratio_x).clip(0, self.target_image_shape[1])
            boxes[   :, 1] = (boxes[   :, 1] / ratio_y).clip(0, self.target_image_shape[0])

        boxes = (boxes + .5).astype(np.int32)
        return boxes

    # data is ImageLabel{image,[Label]}
    def process(self, image_labels):

        # adjust the coordination
        shape = image_labels.image.shape[:2] # h,w
        boxes = self._adjust_by_size(image_labels.bboxes,shape)

        # find the one bbox boundary
        xmins = boxes[:, :, 0].min(axis=1)
        xmaxs = np.maximum(boxes[:, :, 0].max(axis=1), xmins + 1) # 找到x最大值
        ymins = boxes[:, :, 1].min(axis=1)
        ymaxs = np.maximum(boxes[:, :, 1].max(axis=1), ymins + 1)

        character_segment = self.render_character_segemention(image_labels)
        localization_map = np.zeros(self.target_image_shape, dtype=np.float32)
        order_maps = np.zeros((*self.target_image_shape,self.max_sequence), dtype=np.float32)

        assert xmins.shape[0] <= self.max_sequence, \
            f"the train/validate label text length[{len(image_labels.labels)}] must be less than pre-defined max sequence length[{self.max_sequence}]"

        # process each character
        for i in range(xmins.shape[0]):

            # Y_k is the normalized_gaussian map, comply with the name in the paper
            Y_k = self.gaussian_normalize(self.target_image_shape,xmins[i], xmaxs[i], ymins[i], ymaxs[i])

            self.render_order_map(order_maps[:,:,i],Y_k,threshold=self.ζ)
            self.render_localization_map(localization_map,Y_k)

        return character_segment,order_maps,localization_map

    # 围绕中心点做一个高斯分布，但是由于每个点的概率值过小，所以要做一个归一化,使得每个点的值归一化到[0,1]之间
    # Make a gaussian distribution with the center, and do normalization
    def gaussian_normalize(self, shape, xmin, xmax, ymin, ymax):
        out = np.zeros(shape)
        h, w = shape[:2]
        # find the "Center" of polygon
        y = (ymax+ymin+1)//2
        x = (xmax+xmin+1)//2
        if not (w > x and h > y):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("标注超出图像范围，生成高斯样本失败：要求w>x,h>y(w:%f,x:%f,h:%f,y:%f)", w,x,h,y)
            return None

        # prepare the gaussian distribution,refer to paper <<Label Generation>>
        out[y, x] = 1.
        out = fi.gaussian_filter(out, (self.δ, self.δ),output=out, mode='mirror')
        out = out / out.max()
        return out

    def render_order_map(self, order_maps, Y_k, threshold):
        order_maps[:] = Y_k[:]
        order_maps[Y_k < threshold] = 0

    # fill the shrunk zone with the value of character ID
    def render_character_segemention(self, image_labels):
        image_original_shape = image_labels.image.shape[:2]  # h,w

        character_segment = np.zeros(self.target_image_shape, dtype=np.int32)
        for one_word_label in image_labels.labels:
            poly = self._adjust_by_size(one_word_label.bbox,image_original_shape)
            label = one_word_label.label
            shrinked_poly = image_utils.shrink_poly(poly,self.shrink)
            char_id = label_utils.str2id(label,self.charset)

            word_fill = np.zeros(self.target_image_shape,np.uint32)
            word_fill.fill(char_id)

            mask = np.zeros(self.target_image_shape, np.uint8)

            cv2.fillPoly(mask, [shrinked_poly], 1)
            mask = np.array(mask)

            character_segment+= mask * word_fill
            character_segment.astype(np.int32)

        return character_segment

    # merge all Y_k to
    def render_localization_map(self,localization_map, Y_k):
        localization_map+=Y_k