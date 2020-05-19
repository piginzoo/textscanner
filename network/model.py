from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from network.layers.fcn_layer import FCNLayer
from network.layers.class_branch_layer import ClassBranchLayer
from network.layers.geometry_branch_layer import GeometryBranch
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import backend as K
from utils.util import call_debug as _call
import logging

logger = logging.getLogger(__name__)

HUBER_DELTA = 0.5


def localization_map_loss():
    def smoothL1(y_true, y_pred):
        x = K.abs(y_true - y_pred)
        x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        return K.sum(x)

    return smoothL1


class TextScannerModel(Model):
    """
        TextScanner Core Model
    """

    def __init__(self, conf, charset):
        super(TextScannerModel, self).__init__()
        self.input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH, 3), name='input_image')
        self.class_branch = ClassBranchLayer(name="ClassBranchLayer", charset_size=len(charset),
                                             filter_num=conf.FILTER_NUM)
        self.geometry_branch = GeometryBranch(name="GeometryBranchLayer", conf=conf)
        # self.word_formation = WordFormation()
        # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
        self.resnet50_model = ResNet50(include_top=False, weights='imagenet')
        self.fcn = FCNLayer(name="FCNLayer", resnet50_model=self.resnet50_model)

    def call(self, inputs, training=None):
        fcn_features = _call(self.fcn, inputs)
        charactor_segmantation = _call(self.class_branch, fcn_features)
        order_map, localization_map = _call(self.geometry_branch, fcn_features)
        # word = self.word_formation(charactor_segmantation,order_map)
        return charactor_segmantation, order_map, localization_map
