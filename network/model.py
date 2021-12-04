from network.layers.class_branch_layer import ClassBranchLayer
from network.layers.geometry_branch_layer import GeometryBranch
from network.layers.word_formation_layer import WordFormation
from tensorflow.keras.applications.resnet import ResNet50
from network.layers.fcn_layer import FCNLayer
from tensorflow.keras.optimizers import Adam
from utils.logger import call_debug as _call
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import logging

logger = logging.getLogger(__name__)

HUBER_DELTA = 0.5


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
        self.resnet50_model = ResNet50(include_top=False,
                                       weights='imagenet')  # Resnet50+FCN：参考 http://www.piginzoo.com/machine-learning/2020/04/23/fcn-unet#resnet50%E7%9A%84fcn
        self.resnet50_model.summary()
        self.fcn = FCNLayer(name="FCNLayer", filter_num=conf.FILTER_NUM, resnet50_model=self.resnet50_model)

    def call(self, inputs, training=None):
        fcn_features = _call(self.fcn, inputs)
        character_segmentation = _call(self.class_branch, fcn_features)

        localization_map, order_segmentation = _call(self.geometry_branch, fcn_features)

        return {'character_segmentation': character_segmentation,
                'order_segmentation': order_segmentation,
                'localization_map': localization_map}

    def localization_map_loss(self):
        def smoothL1(y_true, y_pred):
            x = K.abs(y_true - y_pred)
            x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return K.sum(x)

        return smoothL1

    def comile_model(self):
        # Lc, Lo, Ls, Lm are the losses for:(refer from paper)
        # - localization map,
        # - order segmentation,
        # - text segmentation and
        # - mutual supervision loss
        losses = {'character_segmentation': 'categorical_crossentropy',
                  'order_segmentation': 'categorical_crossentropy',
                  'localization_map': self.localization_map_loss()}
        loss_weights = {'character_segmentation': 1,
                        'order_segmentation': 10,
                        'localization_map': 10}
        metrics = {'character_segmentation':['categorical_accuracy'],
                   'order_segmentation':['categorical_accuracy'],
                   'localization_map':['binary_accuracy']}

        self.compile(Adam(),
                     loss=losses,
                     loss_weights=loss_weights,
                     metrics=metrics,
                     run_eagerly=True)

        self.build(self.input_image.shape)  # no build, no summary
        logger.info("######## TextScanner Model Structure ########")
        self.summary()
        logger.info("TextScanner Model was compiled.")
