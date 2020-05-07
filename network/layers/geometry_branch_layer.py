from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import GRU
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

    def build(self, input_shape):
        # order segment generation network
        self.conv_os1 = Convolution2D(filters=512,kernel_size=(3,3),strides=2) # 1/2
        self.conv_os2 = Convolution2D(filters=512,kernel_size=(3, 3), strides=2)  # 1/4
        self.conv_os3 = Convolution2D(filters=512,kernel_size=(3, 3), strides=2)  # 1/8
        self.gru_os = GRU(units=64)
        self.dconv_os1 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1/4
        self.dconv_os2 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1/2
        self.dconv_os3 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1

        # localization map generation network
        self.conv_lm1 = Convolution2D(512,(3,3),64)
        self.conv_lm2 = Convolution2D(512,(1,1),self.sequence_length)

    def call(self, inputs, training=None):
        # generate Order Segmentation
        s1 = self.conv_os1(inputs)
        s2 = self.conv_os2(s1)
        s3 = self.conv_os3(s2)

        # [B,H,W,C] => [B,W,H*C]
        s = tf.transpose(s3,(0,2,1,3))
        shape = s.get_shape().as_list()
        print("==============================")
        print(shape)
        s = tf.reshape(-1,self.image_area)

        # pass a GRU
        s = self.gru_os(s)

        s = self.dconv_os3(s)
        s = self.dconv_os2(s+s2)
        s = self.dconv_os1(s+s1)

        # generate Localization Map
        q = self.conv_lm1(inputs)
        q = self.conv_lm2(q)

        # multiplq S[N,H,W] * Q[1,H,W] =>  [N,H,W]
        h = s*q
        return h,q