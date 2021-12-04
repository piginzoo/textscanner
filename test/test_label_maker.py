from utils.label.label_maker import LabelGenerater
from utils.label.label import ImageLabel
from utils.label import label_utils
import matplotlib.pyplot as plt
import numpy as np
import logging
import os, cv2
import time
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


def save_bbox_image(image_label, image_path):
    image = image_label.image
    bboxes = image_label.bboxes
    cv2.polylines(image,bboxes,True,(0,0,255))
    cv2.imwrite(image_path,image)


def save_image(name, gt, image=None, highlight=False):
    image = cv2.resize(image, shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if highlight:
        gt_mask = gt.copy()
        gt_mask[gt_mask > 0] = 1
        gt = 155 + 100 * gt / (gt.max() + 0.001)
        gt = gt * gt_mask
    else:
        gt = 255 * gt / (gt.max() + 0.001)

    image = np.ubyte(0.3 * gt + 0.7 * image)
    plt.clf()
    # plt.imshow(image)
    plt.imsave(name, image)  # 必须使用plt，才可以显示彩色的

    # cv2.imwrite(name,image)   # 如果使用cv2，出来的都是灰度的，不知道为何，plt对灰度的显示做了特殊的处理，使其彩色花了


def test_make_label(image_path, charset):
    dir, image_name = os.path.split(image_path)
    name, ext = os.path.splitext(image_name)
    if ext != ".png" and ext != ".jpg":
        # print("[ERROR] 不是图像：", image_name)
        return None,None
    json_path = os.path.join(dir, name + ".txt")

    print("----------------------------------------------")
    print("Image: ", image_name)

    start = time.time()
    image = cv2.imread(image_path)
    f = open(json_path, encoding="utf-8")
    data = f.readlines()
    image_label = ImageLabel(image,
                             data,
                             format="plaintext",
                             target_size=(conf.INPUT_IMAGE_WIDTH, conf.INPUT_IMAGE_HEIGHT))

    generator = LabelGenerater(conf.MAX_SEQUENCE,
                               target_image_shape=(conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH),
                               charset=charset)
    character_segment, order_segments, order_maps, localization_map = generator.process(image_label)
    time_eclapse_makelabel = time.time() - start

    if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    save_bbox_image(image_label, os.path.join(debug_dir,f"{name}.jpg"))
    save_image(os.path.join(debug_dir, f"{name}_character_segment.jpg"), character_segment, image, True)
    save_image(os.path.join(debug_dir, f"{name}_localization_map.jpg"), localization_map, image)
    # order_maps = order_maps.transpose(2, 0, 1)  # (H,W,S) => (S,H,W)

    for i in range(order_maps.shape[-1]):# [H,W,S]
        save_image(os.path.join(debug_dir, f"{name}_order_map_{i + 1}.jpg"), order_maps[:,:,i], image)

    # 尝试还原结果，看看是不是可以在复原判断出原有的汉字，
    # 主要是验证这样识别是不是一个合理的方法（通过标注来尝试，标注理论上应该是最容易得到正确字符的）
    time_eclapse_word_formulation =  word_formulation.process(character_segment, charset, image_label, order_maps)

    return time_eclapse_makelabel, time_eclapse_word_formulation

from utils import word_formulation




# python -m test.test_label_maker
if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.DEBUG)

    charset = label_utils.get_charset(charset_path)

    # test  目录里的所有
    dir = "data/train"
    files = os.listdir(dir)
    time_all_ml = time_all_wf = count = 0
    for f in files:
        image_path = os.path.join(dir,f)
        t_ml, t_wf = test_make_label(image_path, charset)
        if t_ml:
            count+=1
            time_all_ml += t_ml
            time_all_wf += t_wf
            print("此样本处理耗时:%f秒,Word Form: %f" % (t_ml, t_wf))
    print("平均每样本耗时：%f秒, Word Form平均耗时：" %(time_all_ml/count,time_all_wf/count))

    # test 单张
    # time = test_make_label("data/train/3-5.png", charset)
    # print("耗时:%d秒" % time)
    # test_make_label("data/train/0-6.png", charset)
    # test_make_label("data/train/0-23.png", charset)
    # test_make_label("data/train/2-16.png", charset)
    # test_make_label("data/train/1-22.png", charset)
