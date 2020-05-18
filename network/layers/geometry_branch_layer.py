from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Activation
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

    def build(self, input_shape,FILTER_NUM=4):
        # order segment generation network
        self.conv_order_seg1 = Convolution2D(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="conv_order_seg1", padding="same") # 1/2
        self.conv_order_seg2 = Convolution2D(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="conv_order_seg2", padding="same")  # 1/4
        self.conv_order_seg3 = Convolution2D(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="conv_order_seg3", padding="same")  # 1/8
        self.transpose1 = Permute((2,1,3)) # [B,H,W,C] => [B,W,H,C]
        # self.reshape1 = Reshape((-1,self.conf.INPUT_IMAGE_WIDTH,self.conf.INPUT_IMAGE_HEIGHT*FILTER_NUM)) # [B,W,H,C] => [B,W,H*C]
        self.gru_order_seg = GRU(units=FILTER_NUM * (input_shape[1]//8), return_sequences=True, name="gru_order_seg")
        # self.reshape2 = Reshape((-1,self.conf.INPUT_IMAGE_WIDTH,self.conf.INPUT_IMAGE_HEIGHT,FILTER_NUM)) # [B,W,H*C] => [B,W,H,C]
        self.transpose2 = Permute((2,1,3)) # [B,W,H,C] => [B,H,W,C]
        self.dconv_order_seg1 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, name="dconv_order_seg1", padding="same")  # 1/4
        self.dconv_order_seg2 = Conv2DTranspose(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="dconv_order_seg2", padding="same")  # 1/2
        self.dconv_order_seg3 = Conv2DTranspose(filters=FILTER_NUM, kernel_size=(3, 3), strides=2, name="dconv_order_seg3", padding="same")  # 1
        self.sigmoid = Activation("sigmoid", name="sigmoid")

        # localization map generation network
        self.conv_loc_map1 = Convolution2D(filters=FILTER_NUM,           kernel_size=(3,3), padding="same", name="conv_loc_map1")
        self.conv_loc_map2 = Convolution2D(filters=self.sequence_length, kernel_size=(1,1), padding="same", name="conv_loc_map2")
        self.softmax = Softmax(name="softmax")


    def call(self, inputs, training=None):
        # convs
        x = inputs
        x = s1 = _call(self.conv_order_seg1,x)
        x = s2 = _call(self.conv_order_seg2,x)
        x = _call(self.conv_order_seg3,x)

        # gru
        x = _call(self.transpose1,x)
        height = x.shape[2]
        channel = x.shape[3]
        target_shape = [-1, x.shape[1], height*channel]
        x = _call(tf.reshape, x, target_shape)
        x = _call(self.gru_order_seg, x)
        target_shape = [-1, x.shape[1], height, channel]
        x = _call(tf.reshape, x, target_shape)
        x = _call(self.transpose2, x)

        # de-convs
        x = _call(self.dconv_order_seg3, x)
        x = _call(self.dconv_order_seg2, x+s2)
        x = _call(self.dconv_order_seg1, x+s1)
        s = _call(self.sigmoid, x)

        # generate Localization Map
        q = _call(self.conv_loc_map1, inputs)
        q = _call(self.conv_loc_map2, q)
        q = _call(self.softmax, q)

        # multiply S[B,H,W,N] * Q[B,H,W,1] => [B,H,W,N]
        h = s*q

        return h, q