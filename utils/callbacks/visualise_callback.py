from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.callbacks import Callback
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import io, cv2
import conf

logger = logging.getLogger(__name__)


class VisualCallback(Callback):
    """
        Visualization the training process for debugging
    """

    def __init__(self, tag, tboard_dir, image_loader, debug_step):
        super().__init__()
        self.tag = tag
        self.debug_step = debug_step
        self.tboard_dir = tboard_dir
        self.font = ImageFont.truetype("data/fonts/simsun.ttc", 10)  # 设置字体
        self.image_loader = image_loader

    def on_batch_end(self, batch, logs=None):

        if batch % self.debug_step != 0: return

        logger.debug("Try to dump the debug images to tboard")

        self.image_loader.shuffle()
        data = self.image_loader.data_list[:conf.VISUAL_IMAGES]  # hard code 9 images
        images, labels = self.image_loader.load_image_label(data)

        writer = tf.summary.create_file_writer(self.tboard_dir)

        pred = self.model(images)  # return [character_segment, order_map, localization_map]

        # [W,H,C]=>[W,H] by argmax to adapt the image dimension
        label_character_segments = np.argmax(labels['character_segmentation'], axis=-1)
        pred_character_segments = np.argmax(pred['character_segmentation'], axis=-1)
        label_localization_maps = labels['localization_map']
        pred_localization_maps = pred['localization_map']
        label_order_maps = labels['order_map']
        pred_order_maps = pred['order_map']

        for i in range(len(images)):
            image = images[i]

            self.draw_image(writer, f"label_character_segment_{i}", image, label_character_segments[i], highlight=True)
            self.draw_image(writer, f"pred_character_segment_{i}", image, pred_character_segments[i], highlight=True)
            self.draw_image(writer, f"label_localization_map_{i}", image, label_localization_maps[i])
            self.draw_image(writer, f"pred_localization_map_{i}", image, pred_localization_maps[i])
            for j in range(label_order_maps[i].shape[-1]):
                self.draw_image(writer, f"label_order_maps_{i}_{j}", image, label_order_maps[i][:, :, j])
            for j in range(pred_order_maps[i].shape[-1]):
                self.draw_image(writer, f"pred_order_maps_{i}_{j}", image, pred_order_maps[i][:, :, j])

        writer.close()
        logger.debug("Visual Debugging done!")
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
