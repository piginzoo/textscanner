from tensorflow.keras.layers import Layer
import tensorflow

class WordFormation(Layer):
    """
       integral the product of "Character Segmentation" & "Order Maps",
       and infer the character possibility.
       The threshold is ?
    """

    def __init__(self):
        super(WordFormation, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, G, H, training=None):
        # [N,H,W,C]，[N:字符个数(30)，高，宽，C:字库数(3770)]
        # make a integral for all points
        # p_k  =  \int_{(x,y)\in\Omega} G(x,y) * H_k(x,y)
        # 要对Pk做积分，我本来想用tensorflow.contrib.integrate.odeint_fixed，
        # 但是发现这个库似乎不能用了，在tf2.0没有这个定积分函数了，
        # 后来仔细一想，这个实际上就是一个求和操作啊，只是写成这个形式而已，
        # 所以事先相加就可以了
        return tensorflow.math.reduce_sum(G*H)