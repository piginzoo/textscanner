from tensorflow.python.keras.models import Model
from network.layers.fcn_layer import FCNLayer
from network.layers.class_branch_layer import ClassBranchLayer
from network.layers.geometry_branch_layer import GeometryBranch
from network.layers.word_formation_layer import WordFormation
import logging

logger = logging.getLogger(__name__)


class TextScannerModel(Model):
    """
        TextScanner Core Model
    """
    def __init__(self):
        super(TextScannerModel, self).__init__()
        self.fcn = FCNLayer(28 * 28, 256)
        self.class_branch = ClassBranchLayer(256, 128)
        self.geometry_branch = GeometryBranch(128, 64)
        self.word_formation = WordFormation(188,188)

    def call(self, inputs, training=None):
        x = self.fcn(inputs)
        charactor_segmantation = self.class_branch(x)
        order_map = self.geometry_branch(x)
        result = self.word_formation(charactor_segmantation,order_map)
        return result