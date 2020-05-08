from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class FCNLayer(Layer):
    """
    # Resnet层的参考图：http://www.piginzoo.com/machine-learning/2019/08/28/east &  https://i.stack.imgur.com/tkUYS.png
    # FC实现参考：https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py
    # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
    """

    def __init__(self,resnet50_model):
        super(FCNLayer, self).__init__()
        resnet50_model.layers.pop()
        self.resnet50_model = resnet50_model


    # 谁小，按照谁的大小切
    def crop(self, o1, o2):
        output_height1 = o1.shape[0]
        output_width1 = o1.shape[1]
        output_height2 = o2.shape[0]
        output_width2 = o2.shape[1]

        cx = abs(output_width1 - output_width2)
        cy = abs(output_height2 - output_height1)

        if output_width1 > output_width2:
            # cropping=((top_crop, bottom_crop), (left_crop, right_crop))`
            o1 = Cropping2D(cropping=((0, 0),  (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0),  (0, cx)))(o2)

        if output_height1 > output_height2:
            o1 = Cropping2D(cropping=((0, cy),  (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy),  (0, 0)))(o2)

        return o1, o2

    # 实现了fcn_8,也就是结合了pool3,pool4,pool5的感受尺度最多的方式。
    # 由于经过resnet50后的尺寸是2x8，所以，kernel设计成2x2，不再是大家普通用的7x7
    def call(self,input_image,n_classes=1024):

        # 获取Resnet50的pool3-pool5,pool3-5的命名是参照FCN中的定义，"convx_blockx_out"是使用HDFView查看hdf5模型后找到的
        # pool5 = self.resnet50_model(input_image)
        pool3 = self.resnet50_model.get_layer("conv3_block4_out").output # 1/8，[28,28.512]
        pool4 = self.resnet50_model.get_layer("conv4_block6_out").output # 1/16 [14,14,1024]
        pool5 = self.resnet50_model.get_layer("conv5_block3_out").output # 1/32 [7,7,2048]

        # pool5经过2x2,1x1,1x1的卷积后，做反卷积，恢复到原图1/16
        o = pool5
        o = (Conv2D(4096, (2, 2), activation='relu',padding='same'))(o)
        o = Dropout(0.5)(o)
        o = (Conv2D(4096, (1, 1), activation='relu',padding='same'))(o)
        o = Dropout(0.5)(o)
        o = (Conv2D(512,  (1, 1), kernel_initializer='he_normal',))(o)
        # stride=2后，反卷积图从2x8=>5x17（像素间padding0），采用3x3核做上卷积
        o = Conv2DTranspose(filters=n_classes, kernel_size=(3, 3),  strides=(2, 2), use_bias=False)(o)

        # pool4做1x1卷积后 + 反卷积后的pool5，恢复到原图1/16
        o2 = pool4
        o2 = (Conv2D(filters=1024,kernel_size=(1, 1), kernel_initializer='he_normal',))(o2)

        # o, o2 = self.crop(o, o2) # 剪裁到原图大小
        o = Add()([o, o2])

        # （pool4 + 上采样后的pool5）的结果 再次做反卷积，尺寸恢复到原图的1/8
        o = Conv2DTranspose(filters=n_classes, kernel_size=(3, 3),  strides=(2, 2), use_bias=False)(o)

        # pool3做1x1卷积后与上面的结果融合
        o2 = pool3
        o2 = (Conv2D(filters=n_classes,  kernel_size=(1, 1), kernel_initializer='he_normal'))(o2)
        # o2, o = self.crop(o2, o)
        o = Add()([o2, o])

        # 最后一个反卷积，将尺寸从1/8，直接恢复到原图大小（stride=8)
        o = Conv2DTranspose(filters=n_classes, kernel_size=(3, 3),  strides=(8, 8), use_bias=False)(o)

        return o
