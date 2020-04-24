from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import GRU
import tensorflow as tf

from tensorflow.nn import softmax

class GeometryBranch(Layer):
    """
    [H,W,E] => [H,W,C]
    E: encoding output channels
    C: character class number
    """
    def __init__(self,sequence_length=32):
        super(GeometryBranch, self).__init__()
        self.sequence_length = sequence_length

    def build(self, input_shape):
        self.conv_os1 = Convolution2D(filter=(3,3),kernel_size=16,strides=2) # 1/2
        self.conv_os2 = Convolution2D(filter=(3, 3), kernel_size=16, strides=2)  # 1/4
        self.conv_os3 = Convolution2D(filter=(3, 3), kernel_size=16, strides=2)  # 1/8
        self.gru_os = GRU(units=64)
        self.dconv_os1 = Conv2DTranspose(filter=(3, 3), kernel_size=16, strides=2)  # 1/4
        self.dconv_os2 = Conv2DTranspose(filter=(3, 3), kernel_size=16, strides=2)  # 1/2
        self.dconv_os3 = Conv2DTranspose(filter=(3, 3), kernel_size=16, strides=2)  # 1

        self.conv_lm1 = Convolution2D((3,3),64)
        self.conv_lm2 = Convolution2D((1,1),self.sequence_length)

    def call(self, inputs, training=None):

        # generate Order Segmentation
        x1 = self.conv_os1(inputs)
        x2 = self.conv_os2(x1)
        x3 = self.conv_os3(x2)
        # [H,W,C] => [W,H*C]
        x = tf.transpose(x3,(-1,x3.shape[]))
        x = self.gru_os(x)
        x = self.dconv_os3(x)
        x = self.dconv_os2(x+x2)
        x = self.dconv_os1(x+x1)

        # generate Localization Map
        y = self.conv_lm1(inputs)
        y = self.conv_lm2(y)

        r = x*y

        return r