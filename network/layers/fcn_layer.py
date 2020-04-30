from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.python.keras.applications.resnet import ResNet50
from network.fcn.resnet50 import get_resnet50_encoder


class FCNLayer(Layer):
    """
    # Resnet层的参考图：http://www.piginzoo.com/machine-learning/2019/08/28/east &  https://i.stack.imgur.com/tkUYS.png
    # FC实现参考：https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py
    # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
    """

    # crop o1 wrt o2
    def crop(self,o1, o2, i):
        o_shape2 = Model(i, o2).output_shape

        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

        o_shape1 = Model(i, o1).output_shape
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

        cx = abs(output_width1 - output_width2)
        cy = abs(output_height2 - output_height1)

        if output_width1 > output_width2:
            o1 = Cropping2D(cropping=((0, 0),  (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0),  (0, cx)))(o2)

        if output_height1 > output_height2:
            o1 = Cropping2D(cropping=((0, cy),  (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy),  (0, 0)))(o2)

        return o1, o2

    # 我直接实现了fcn_8,
    def fcn_8(self,n_classes):
        # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
        resnet50_model = ResNet50(include_top=False,weights='imagenet',input_shape=(32,256,3))
        input_image = resnet50_model.input

        # 获取Resnet50的pool3-pool5,
        # pool3-5的命名是参照FCN中的定义，
        pool3 = resnet50_model.get_layer("activate_22") # 1/8
        pool4 = resnet50_model.get_layer("activate_40") # 1/16
        pool5 = resnet50_model.get_layer("activate_49") # 1/32

        #
        o = pool5
        o = (Conv2D(4096, (7, 7), activation='relu',padding='same'))(o)
        o = Dropout(0.5)(o)
        o = (Conv2D(4096, (1, 1), activation='relu',padding='same'))(o)
        o = Dropout(0.5)(o)
        o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',))(o)
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(o)

        o2 = pool4
        o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',))(o2)
        o, o2 = self.crop(o, o2, input_image)
        o = Add()([o, o2])
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(o)

        o2 = pool3
        o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal'))(o2)
        o2, o = self.crop(o2, o, img_input)
        o = Add()([o2, o])

        o = Conv2DTranspose(n_classes, kernel_size=(16, 16),  strides=(8, 8), use_bias=False)(o)

        model = get_segmentation_model(img_input, o)
        model.model_name = "fcn_8"
        return inputs