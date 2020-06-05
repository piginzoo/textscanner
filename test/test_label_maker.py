from utils.label.label_maker import LabelGenerater
from utils.label.label import ImageLabel
from utils.label import label_utils
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import conf

debug_dir = "data/debug"
charset_path = "config/charset.4100.txt"
shape = (conf.INPUT_IMAGE_WIDTH, conf.INPUT_IMAGE_HEIGHT)

"""
    这个类用于测试样本生成，主要测试：
    1、是不是从视觉上看，符合要求：
        - order_segment是不是画出来，恰好是包裹字符的
        - localization map是不是围绕字符中心的一个正态分布的样子
        - order map是不是按照顺序画出了第N个字符的正态分布
    2. 测试是不是按照word formulation，可以从生成的label得到原有的字符串，
       这样变相的验证了，生成的各种map的正确性
"""


def save_image(name, gt, image, highlight=False):
    image = cv2.resize(image, shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if highlight:
        gt_mask = gt.copy()
        gt_mask[gt_mask > 0] = 1
        gt = 155 + 100 * gt / (gt.max() + 0.001)
        gt = gt * gt_mask
    else:
        gt = 255 * gt / (gt.max() + 0.001)


    image = np.ubyte(0.7 * gt + 0.3 * image)
    plt.clf()
    # plt.imshow(image)
    plt.imsave(name, image)  # 必须使用plt，才可以显示彩色的

    # cv2.imwrite(name,image)   # 如果使用cv2，出来的都是灰度的，不知道为何，plt对灰度的显示做了特殊的处理，使其彩色花了


def test_make_label(image_path, json_path, name, charset):
    image = cv2.imread(image_path)

    f = open(json_path, encoding="utf-8")
    data = f.readlines()

    image_label = ImageLabel(image,
                             data,
                             format="plaintext",
                             target_size=(256, 64))

    generator = LabelGenerater(conf.MAX_SEQUENCE,
                               target_image_shape=(conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH),
                               charset=charset)

    character_segment, order_maps, localization_map = generator.process(image_label)

    if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    save_image(os.path.join(debug_dir, f"character_segment_{name}.jpg"), character_segment, image, True)
    save_image(os.path.join(debug_dir, f"localization_map_{name}.jpg"), localization_map, image)
    order_maps = order_maps.transpose(2, 0, 1)  # (H,W,S) => (S,H,W)

    for i, order_map in enumerate(order_maps):
        save_image(os.path.join(debug_dir, f"order_map_{name}_{i + 1}.jpg"), order_map, image)

    test_word_formulation(character_segment, charset, image_label, order_maps)


# 尝试还原结果，看看是不是可以在复原判断出原有的汉字，
# 主要是验证这样识别是不是一个合理的方法（通过标注来尝试，标注理论上应该是最容易得到正确字符的）
def test_word_formulation(character_segment_G, charset, image_label, order_maps_H):
    G = np.eye(len(charset))[character_segment_G]  # eye是对角阵生成函数，通过他，完成categorical one hot化
    H = order_maps_H
    print("character_segment_G.shape:", character_segment_G.shape)
    print("G.shape:", G.shape)
    print("order_maps.shape/H:", order_maps_H.shape)

    pred = ""
    for i, H_k in enumerate(H):
        # G[H,W,C:4100] * H_k[H,W,1]
        # G是每个像素字符的概率(1/4100)
        # H_k是第k个字符对应的正态分布
        # (G*H_k) ===> [H,W,4100]
        # sum = \sum(G*H_k) ===> [4100]

        _H_k = H_k[:, :, np.newaxis]  # [H,W] => [H,W,1]
        GH_k = (G * _H_k)
        print("GH_k max value:",GH_k.max())

        GH_k = np.sum(GH_k, axis=0)

        sum = np.sum(GH_k, axis=0)

        id = sum.argmax()

        print("max id of 4100:", id, ", max value is :", sum[id])
        if id == 0: continue

        c = label_utils.id2str([int(id)], charset)
        pred += c

    print("Label  :[%s]" % image_label.label)
    print("Predict:[%s]" % pred)


if __name__ == "__main__":
    charset = label_utils.get_charset(charset_path)

    import logging

    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.DEBUG)

    # test  目录里的所有
    # dir = "data/train"
    # files = os.listdir(dir)
    # for f in files:
    #     name, ext = os.path.splitext(f)
    #     image_path = os.path.join(dir,f)
    #     if ext!= ".png": continue
    #     label_path = os.path.join(dir,name+".txt")
    #     print("----------------------------------------------")
    #     print("Image: ",name)
    #     test_make_label(image_path, label_path, name, charset)


    # test 单张
    test_make_label("data/train/1-8.png", "data/train/1-8.txt", "1-8", charset)
    # test_make_label("data/train/1-1.png", "data/train/1-1.txt", "1-1", charset)
