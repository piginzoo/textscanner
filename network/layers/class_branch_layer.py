from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.keras.activations import softmax


class ClassBranchLayer(Layer):
    """
    [H,W,E] => [H,W,C]
    E: encoding output channels
    C: character class number
    """
    def __init__(self,charset_size=0):
        super(ClassBranchLayer, self).__init__()
        self.charset_size = charset_size

    def build(self, input_shape):
        self.conv1 = Convolution2D((3,3),512)
        self.conv2 = Convolution2D((1,1),self.charset_size)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = softmax(x)
        return x


