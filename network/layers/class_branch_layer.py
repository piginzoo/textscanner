from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Softmax
from utils.util import call_debug as _call


class ClassBranchLayer(Layer):
    """
    [H,W,E] => [H,W,C]
    E: encoding output channels
    C: character class number
    """

    def __init__(self, name, charset_size, filter_num):
        super().__init__(name=name)
        self.charset_size = charset_size
        self.filter_num = filter_num

    def build(self, input_shape):
        self.conv1 = Convolution2D(filters=self.filter_num, kernel_size=(3, 3), padding="same",
                                   name="class_branch_conv1")
        # the classification number is Character Size + 1
        self.conv2 = Convolution2D(filters=self.charset_size + 1, kernel_size=(1, 1), padding="same",
                                   name="class_branch_conv2")
        self.softmax = Softmax(name="class_branch_softmax")

    def call(self, inputs, training=None):
        x = _call(self.conv1, inputs)
        x = _call(self.conv2, x)
        x = _call(self.softmax, x)
        return x
