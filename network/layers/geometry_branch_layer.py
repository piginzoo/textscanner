from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.layers import Permute
from tensorflow.python.keras.layers import Reshape
from utils.util import call_debug as _call
import tensorflow as tf


class GeometryBranch(Layer):
    """
    [H,W,E] => [H,W,C]
    E: encoding output channels
    C: character class number
    """
    def __init__(self,conf):
        super(GeometryBranch, self).__init__()
        self.image_area = conf.INPUT_IMAGE_HEIGHT * conf.INPUT_IMAGE_WIDTH
        self.sequence_length = conf.MAX_SEQUENCE
        self.conf = conf

    def build(self, input_shape,FILTER_NUM=512):
        # order segment generation network
        self.conv_order_seg1 = Convolution2D(filters=FILTER_NUM, kernel_size=(3,3),strides=2, name="conv_order_seg1") # 1/2
        self.conv_order_seg2 = Convolution2D(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="conv_order_seg2")  # 1/4
        self.conv_order_seg3 = Convolution2D(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="conv_order_seg3")  # 1/8
        self.transpose1 = Permute((2,1,3)) # [B,H,W,C] => [B,W,H,C]
        # self.reshape1 = Reshape((-1,self.conf.INPUT_IMAGE_WIDTH,self.conf.INPUT_IMAGE_HEIGHT*FILTER_NUM)) # [B,W,H,C] => [B,W,H*C]
        self.gru_order_seg = GRU(units=FILTER_NUM, return_sequences=True, name="gru_order_seg")
        # self.reshape2 = Reshape((-1,self.conf.INPUT_IMAGE_WIDTH,self.conf.INPUT_IMAGE_HEIGHT,FILTER_NUM)) # [B,W,H*C] => [B,W,H,C]
        self.transpose2 = Permute((2,1,3)) # [B,W,H,C] => [B,H,W,C]
        self.dconv_order_seg1 = Conv2DTranspose(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="dconv_order_seg1")  # 1/4
        self.dconv_order_seg2 = Conv2DTranspose(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="dconv_order_seg2")  # 1/2
        self.dconv_order_seg3 = Conv2DTranspose(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="dconv_order_seg3")  # 1

        # localization map generation network
        self.conv_loc_map1 = Convolution2D(FILTER_NUM,(3,3),64)
        self.conv_loc_map2 = Convolution2D(FILTER_NUM,(1,1),self.sequence_length)


    def call(self, inputs, training=None):
        # 1.generate Order Segmentation
        # 1.1 conv
        x = inputs
        x = s1 = _call(self.conv_order_seg1,x)
        x = s2 = _call(self.conv_order_seg2,x)
        x = _call(self.conv_order_seg3,x)
        x = _call(self.transpose1,x)
        height = x.shape[2]
        channel = x.shape[3]
        target_shape = [-1, x.shape[1], height*channel]
        print(target_shape)
        x = tf.reshape(x,target_shape)
        # x = _call(self.reshape1, x)
        x = _call(self.gru_order_seg,x)
        target_shape = [-1, x.shape[1], height, channel]
        # x = tf.reshape(x,target_shape)
        x = tf.reshape(x, target_shape)
        x = _call(self.transpose2,x)

        # 1.4 deconv
        x = _call(self.dconv_order_seg3,x)
        x = _call(self.dconv_order_seg2,x+s2)
        s = _call(self.dconv_order_seg1,x+s1)

        # 2.generate Localization Map
        q = _call(self.conv_loc_map1,inputs)
        q = _call(self.conv_loc_map2,q)

        # multiply S[N,H,W] * Q[1,H,W] =>  [N,H,W]
        h = s*q

        return h,q