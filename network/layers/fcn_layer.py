from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Cropping2D, Layer
from tensorflow.keras.models import Model
from utils.logger import call_debug as _call


class FCNLayer(Layer):
    """
    # Resnet：http://www.piginzoo.com/machine-learning/2019/08/28/east &  https://i.stack.imgur.com/tkUYS.png
    # FCN：   https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py
    # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
    This implements FCN-8s
    """

    def __init__(self, name, filter_num, resnet50_model):
        super().__init__(name=name)
        resnet50_model.layers.pop()
        # resnet50_model.summary()
        self.resnet50_model = resnet50_model
        self.filter_num = filter_num

    def build(self, input_image):

        ############################
        # encoder part
        ############################

        layer_names = [
            "conv3_block4_out",  # 1/8
            "conv4_block6_out",  # 1/16
            "conv5_block3_out",  # 1/32
        ]
        layers = [self.resnet50_model.get_layer(name).output for name in layer_names]
        self.FCN_left = Model(inputs=self.resnet50_model.input, outputs=layers)

        ############################
        # decoder part
        ############################

        # pool5(1/32) ==> 1/16
        self.pool5_conv1 = Conv2D(filters=self.filter_num,
                                  kernel_size=(2, 2),
                                  activation='relu',
                                  padding='same',
                                  name="fcn_pool5_conv1")  # 2x2 is because the least height is 2 pixes after Resnet
        self.pool5_drop1 = Dropout(0.25, name="fcn_pool5_drop1")
        self.pool5_conv2 = Conv2D(filters=self.filter_num,
                                  kernel_size=(1, 1),
                                  activation='relu',
                                  padding='same',
                                  name="fcn_pool5_conv2")
        self.pool5_drop2 = Dropout(0.25, name="fcn_pool5_drop2")
        self.pool5_conv3 = Conv2D(filters=self.filter_num,
                                  kernel_size=(1, 1),
                                  kernel_initializer='he_normal',
                                  name="fcn_pool5_conv3")
        self.pool5_dconv1 = Conv2DTranspose(filters=self.filter_num,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            use_bias=False,
                                            name="fcn_pool5_dconv1")  # stride=2后，反卷积图从2x8=>5x17（像素间padding0），采用3x3核做上卷积

        # pool4(1/16)+dconv ==> 1/8
        self.pool4_conv1 = Conv2D(filters=self.filter_num,
                                  kernel_size=(1, 1),
                                  kernel_initializer='he_normal',
                                  name="fcn_pool4_conv1")  # pool4做1x1卷积后 + 反卷积后的pool5，恢复到原图1/16
        self.pool4_add1 = Add(name="fcn_pool4_add1")
        self.pool4_dconv1 = Conv2DTranspose(filters=self.filter_num,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            use_bias=False,
                                            name="fcn_pool4_dconv1")  # （pool4 + 上采样后的pool5）的结果 再次做反卷积，尺寸恢复到原图的1/8

        # pool3(1/8)+dconv ==> original size
        self.pool3_conv1 = Conv2D(filters=self.filter_num,
                                  kernel_size=(1, 1),
                                  kernel_initializer='he_normal',
                                  name="fcn_pool3_conv1")  # pool3做1x1卷积后与上面的结果融合
        self.pool3_add1 = Add(name="fcn_pool3_add1")
        self.pool3_dconv1 = Conv2DTranspose(filters=self.filter_num,
                                            kernel_size=(3, 3),
                                            strides=(8, 8),
                                            use_bias=False,
                                            name="fcn_pool3_dconv1")  # 最后一个反卷积，将尺寸从1/8，直接恢复到原图大小（stride=8)

    def call(self, input_image, training=True):

        pool3, pool4, pool5 = _call(self.FCN_left, input_image)
        o = _call(self.pool5_conv1, pool5)
        o = _call(self.pool5_drop1, o)
        o = _call(self.pool5_conv2, o)
        o = _call(self.pool5_drop2, o)
        o = _call(self.pool5_conv3, o)
        o5 = _call(self.pool5_dconv1, o)

        o4 = _call(self.pool4_conv1, pool4)
        o5, o4 = self.crop(o5, o4)
        o45 = _call(self.pool4_add1, [o5, o4])
        o45 = _call(self.pool4_dconv1, o45)

        o3 = _call(self.pool3_conv1, pool3)
        o45, o3 = self.crop(o45, o3)
        o = _call(self.pool3_add1, [o45, o3])
        o = _call(self.pool3_dconv1, o)

        return o

    # cut to smaller
    def crop(self, o1, o2):
        o1_height, o1_width = o1.shape[1], o1.shape[2]
        o2_height, o2_width = o2.shape[1], o2.shape[2]

        cx = abs(o1_width - o2_width)
        cy = abs(o1_height - o2_height)

        if o1_width > o2_width:
            o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

        if o1_height > o2_height:
            o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

        return o1, o2
