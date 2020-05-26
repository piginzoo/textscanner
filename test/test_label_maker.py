from label.label_maker import LabelGenerater
from utils import label_utils
from label.label import ImageLabel
import matplotlib.pyplot as plt
import os, cv2
import numpy as np
import conf

debug_dir = "data/debug"
charset_path = "data/charset.txt"
shape = (conf.INPUT_IMAGE_WIDTH, conf.INPUT_IMAGE_HEIGHT)


def save_image(name, gt, image, highlight=False):
    image = cv2.resize(image, shape)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if highlight:
        gt_mask = gt.copy()
        gt_mask[gt_mask > 0] = 1
        gt = 155 + 100 * gt / (gt.max() + 0.001)
        gt = gt * gt_mask
    else:
        gt = 255 * gt / (gt.max() + 0.001)

    image = np.ubyte(0.5 * gt + 0.5 * image)
    plt.clf()
    plt.imshow(image)
    plt.imsave(name,image)      # 必须使用plt，才可以显示彩色的
    # cv2.imwrite(name,image)   # 如果使用cv2，出来的都是灰度的，不知道为何，plt对灰度的显示做了特殊的处理，使其彩色花了



def test_make_label(image_path, json_path, charset):
    image = cv2.imread(image_path)

    f = open(json_path, encoding="utf-8")
    data = f.readlines()

    image_label = ImageLabel(image,
                             data,
                             format="labelme")

    generator = LabelGenerater(conf.MAX_SEQUENCE,
                               target_image_shape=(conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH),
                               charset=charset)

    character_segment, order_maps, localization_map = generator.process(image_label)

    if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    save_image(os.path.join(debug_dir, "character_segment.jpg"), character_segment, image, True)
    save_image(os.path.join(debug_dir, "localization_map.jpg"), localization_map, image)
    order_maps = order_maps.transpose(2,0,1) # (H,W,S) => (S,H,W)

    for i, order_map in enumerate(order_maps):
        save_image(os.path.join(debug_dir, "order_map_{}.jpg".format(i + 1)), order_map, image)


if __name__ == "__main__":
    charset = label_utils.get_charset(charset_path)
    test_make_label("data/test/a1.jpg", "data/test/a1.json", charset)
