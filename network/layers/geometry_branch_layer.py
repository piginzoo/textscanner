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
        self.conv_order_seg1 = Convolution2D(filters=512,kernel_size=(3,3),strides=2) # 1/2
        self.conv_order_seg2 = Convolution2D(filters=512,kernel_size=(3, 3), strides=2)  # 1/4
        self.conv_order_seg3 = Convolution2D(filters=512,kernel_size=(3, 3), strides=2)  # 1/8
        self.gru_order_seg = GRU(units=64,return_sequences = True)
        self.dconv_order_seg1 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1/4
        self.dconv_order_seg2 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1/2
        self.dconv_order_seg3 = Conv2DTranspose(filters=512,kernel_size=(3, 3), strides=2)  # 1

        # localization map generation network
        self.conv_loc_map1 = Convolution2D(512,(3,3),64)
        self.conv_loc_map2 = Convolution2D(512,(1,1),self.sequence_length)

    def call(self, inputs, training=None):
        # 1.generate Order Segmentation
        # 1.1 conv
        s1 = self.conv_order_seg1(inputs)
        s2 = self.conv_order_seg2(s1)
        s3 = self.conv_order_seg3(s2)

        # 1.2 [B,H,W,C] => [B,W,H*C]
        s = tf.transpose(s3,(0,2,1,3))
        s = tf.reshape(tensor=s,shape=(tf.shape(s)[0],tf.shape(s)[1],tf.shape(s)[2]*tf.shape(s)[3]))

        # 1.3 pass a GRU
        s = self.gru_order_seg(s)

        # 1.4 deconv
        s = self.dconv_order_seg3(s)
        s = self.dconv_order_seg2(s+s2)
        s = self.dconv_order_seg1(s+s1)

        # 2.generate Localization Map
        q = self.conv_loc_map1(inputs)
        q = self.conv_loc_map2(q)

        # multiply S[N,H,W] * Q[1,H,W] =>  [N,H,W]
        h = s*q

        return h,q