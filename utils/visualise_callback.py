from tensorflow.keras.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from utils import label_utils
import tensorflow as tf
import numpy as np
import logging
import cv2
import io

logger = logging.getLogger(__name__)


class TBoardVisual(Callback):
    """
        Visualization the training process for debugging
    """

    def __init__(self, tag, tboard_dir, charset, args):
        super().__init__()
        self.tag = tag
        self.args = args
        self.tboard_dir = tboard_dir
        self.charset = charset
        self.font = ImageFont.truetype("data/font/simsun.ttc", 10)  # 设置字体

    def on_batch_end(self, batch, logs=None):

        if batch % self.args.debug_step != 0: return

        # self.validation_data is framework pre-defined variable
        np.random.shuffle(self.validation_data.data_list)
        data = self.validation_data.data_list[:9]  # hard code 9 images
        # images, labels: [batch_cs,batch_om,batch_lm)]
        images, labels = self.validation_data.load_image_label(data)

        writer = tf.summary.FileWriter(self.tboard_dir)

        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            pred = self.model(images[i])  # [character_segment, order_map, localization_map]

            label_character_segment = label[0]
            label_localization_map = label[2]
            label_order_map = label[1]

            pred_character_segment = pred[0]
            pred_localization_map = pred[2]
            pred_order_map = pred[1]
            pred_order_segment = pred[3]

            label = label_utils.prob2str(label_character_segment, self.charset)
            pred = label_utils.prob2str(pred_character_segment, self.charset)

            logger.debug("label字符串:%r", label)
            logger.debug("pred字符串 :%r", pred)

            self.draw(writer, "label_character_segment", image,
                      label_character_segment, label, highlight=True)
            self.draw(writer, "pred_character_segment", image,
                      label_character_segment, pred, highlight=True)

            self.draw(writer, "label_localization_map", image, label_localization_map)
            self.draw(writer, "pred_localization_map", image, pred_localization_map)

            self.draw(writer, "label_order_maps", image, label_order_map)
            self.draw(writer, "pred_order_maps", image, pred_order_map)

            self.draw(writer, "pred_order_segment", image, pred_order_segment)

        writer.close()

        return

    def draw_image(self, writer, name, image, gt_pred, text=None, highlight=False):

        if highlight:  # the color is too shallow, enhance it
            gt_pred_mask = gt_pred.copy()
            gt_pred_mask[gt_pred_mask > 0] = 1
            gt_pred = 155 + 100 * gt_pred / (gt_pred.max() + 0.001)
            gt_pred = gt_pred * gt_pred_mask
        else:
            gt_pred = 255 * gt_pred / (gt_pred.max() + 0.001)

        # we use pyplot, because it can generator colorful image, not gray
        image = np.ubyte(0.5 * gt_pred + 0.5 * image)
        plt.clf()
        image = plt.imshow(image)

        draw = ImageDraw.Draw(image)
        draw.text((2, 2), text, 'red', self.font)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        height, width, channel = image.shape
        tf_img = tf.Summary.Image(height=height,
                                  width=width,
                                  colorspace=channel,
                                  encoded_image_string=image_string)

        summary = tf.Summary(value=[tf.Summary.Value(tag="{}/{}".format(name, name), image=tf_img)])
        writer.add_summary(summary)