from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import CategoricalCrossentropy
from network.layers.fcn_layer import FCNLayer
from tensorflow.keras import backend as K
from network.layers.class_branch_layer import ClassBranchLayer
from network.layers.geometry_branch_layer import GeometryBranch
from network.layers.word_formation_layer import WordFormation

import logging
from tensorflow.python.keras.applications.resnet import ResNet50

logger = logging.getLogger(__name__)


class TextScannerModel(Model):
    """
        TextScanner Core Model
    """
    def __init__(self,conf,charset):
        super(TextScannerModel, self).__init__()
        self.input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH, 3),name='input_image')
        self.class_branch = ClassBranchLayer(len(charset))
        self.geometry_branch = GeometryBranch(conf)
        self.word_formation = WordFormation()        # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
        self.resnet50_model = ResNet50(include_top=False,weights='imagenet')
        self.fcn = FCNLayer(self.resnet50_model)

    def call(self, inputs, training=None):
        x = self.fcn(inputs)
        charactor_segmantation = self.class_branch(x)
        order_map,localization_map = self.geometry_branch(x)
        result = self.word_formation(charactor_segmantation,order_map)
        # data:Y:[batch_cs,batch_om,batch_lm]
        return charactor_segmantation, order_map, localization_map


# 自定义损失函数
class TextScannerLoss(Loss):
    HUBER_DELTA = 0.5
    lambda_l = 10
    lambda_o = 10
    lambda_m = 1# 0:pretrain phase, otherwise 1
    crossentropy = CategoricalCrossentropy()

    def __init__(self):
        super(Loss, self).__init__()

    def smoothL1(self, y_true, y_pred):
        x = K.abs(y_true - y_pred)
        x = K.switch(x < self.HUBER_DELTA, 0.5 * x ** 2, self.HUBER_DELTA * (x - 0.5 * self.HUBER_DELTA))
        return K.sum(x)

    def mutual_loss(self,y_true, y_pred):
        pass

    def call(self, y_true, y_pred):
        y_l_true, y_o_true, y_s_true = y_true
        y_l_pred, y_o_pred, y_s_pred = y_pred

        loss = self.lambda_l * self.smoothL1(y_l_true,y_l_pred) + \
               self.lambda_o * self.crossentropy(y_l_true, y_l_pred) + \
               self.lambda_s * self.crossentropy(y_l_true, y_l_pred) + \
               self.lambda_m * self.mutual_loss(y_l_true, y_l_pred)

        return loss