from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.callbacks import Callback
from utils.data_util import ImageLabelLoader
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import io, cv2

logger = logging.getLogger(__name__)


class TBoardVisual(Callback):
    """
        Visualization the training process for debugging
    """

    def __init__(self, tag, tboard_dir, charset, args, validate_sequence):
        super().__init__()
        self.tag = tag
        self.args = args
        self.tboard_dir = tboard_dir
        self.charset = charset
        self.font = ImageFont.truetype("data/fonts/simsun.ttc", 10)  # 设置字体
        self.validate_sequence = validate_sequence

    def on_batch_end(self, batch, logs=None):

        if batch % self.args.debug_step != 0: return

        logger.debug("Try to dump the debug images to tboard")

        # self.validation_data is framework pre-defined variable
        np.random.shuffle(self.validate_sequence.data_list)
        data = self.validate_sequence.data_list[:9]  # hard code 9 images
        # images, labels: [batch_cs,batch_om,batch_lm)]
        images, labels = self.validate_sequence.image_loader.load_image_label(data)

        writer = tf.summary.create_file_writer(self.tboard_dir)

        # import pdb
        # pdb.set_trace()
        pred = self.model(images)  # return [character_segment, order_map, localization_map]
        logger.debug("Model call,input images:\t%r", images.shape)
        logger.debug("Model call,return character_segment:\t%r", pred[0].shape)
        logger.debug("Model call,return order_map:\t%r", pred[1].shape)
        logger.debug("Model call,return localization_map:\t%r", pred[2].shape)

        label_character_segments = labels[0]
        label_localization_maps = labels[2]
        label_order_maps = labels[1]

        for i in range(len(images)):
            image = images[i]

            label_character_segment = label_character_segments[i]
            label_localization_map = label_localization_maps[i]
            label_order_map = label_order_maps[i]

            pred_character_segment = pred[0][i]
            pred_localization_map = pred[2][i]
            pred_order_map = pred[1][i]
            # pred_order_segment = pred[3]

            logger.debug("label_character_segment:%r", label_character_segment.shape)
            logger.debug("label_localization_map:%r", label_localization_map.shape)
            logger.debug("label_order_map:%r", label_order_map.shape)
            logger.debug("pred_character_segment:%r", pred_character_segment.shape)
            logger.debug("pred_localization_map:%r", pred_localization_map.shape)
            logger.debug("pred_order_map:%r", pred_order_map.shape)

            label_character_segment = np.argmax(label_character_segment, axis=-1)
            pred_character_segment = np.argmax(pred_character_segment, axis=-1)
            label_order_map = np.argmax(label_order_map, axis=-1)
            pred_order_map = np.argmax(pred_order_map, axis=-1)

            self.draw_image(writer, f"label_character_segment_{i}", image, label_character_segment, highlight=True)
            self.draw_image(writer, f"pred_character_segment_{i}", image, pred_character_segment, highlight=True)
            self.draw_image(writer, f"label_localization_map_{i}", image, label_localization_map)
            self.draw_image(writer, f"pred_localization_map_{i}", image, pred_localization_map)
            self.draw_image(writer, f"label_order_maps_{i}", image, label_order_map)
            self.draw_image(writer, f"pred_order_maps_{i}", image, pred_order_map)

            # self.draw(writer, "pred_order_segment", image, pred_order_segment)

        writer.close()

        return

    def draw_image(self, writer, name, image, gt_pred, text=None, highlight=False):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if type(gt_pred) == EagerTensor:
            gt_pred = gt_pred.numpy()

        if highlight:  # the color is too shallow, enhance it
            gt_pred_mask = gt_pred.copy()
            gt_pred_mask[gt_pred_mask > 0] = 1
            gt_pred = 155 + 100 * gt_pred / (gt_pred.max() + 0.001)
            gt_pred = gt_pred * gt_pred_mask
        else:
            gt_pred = 255 * gt_pred / (gt_pred.max() + 0.001)

        # we use pyplot, because it can generator colorful image, not gray
        gt_pred = np.squeeze(gt_pred)
        image = np.ubyte(0.5 * gt_pred + 0.5 * image)  # merge the bbox mask and original image
        plt.clf()  # we use plt, which can help convert GRAY image to colorful
        buffer = io.BytesIO()
        plt.imsave(buffer, image, format='jpg')  # dump the image to buffer
        image = Image.open(buffer).convert('RGB')
        buffer.close()
        image = np.array(image)  # convert from PIL image to ndarray
        image = np.array([image])  # [W,H] => [W,H,1]
        with writer.as_default():
            tf.summary.image(name, image, step=0)
