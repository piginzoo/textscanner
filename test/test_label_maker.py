from label.label_maker import LabelGenerater
from utils import label_utils
from label.label import ImageLabel
import os,cv2
import numpy as np
import conf

debug_dir = "data/debug"
charset_path = "data/charset.txt"
shape = (conf.INPUT_IMAGE_WIDTH,conf.INPUT_IMAGE_HEIGHT)

def save_image(name,gt,image):
    image = cv2.resize(image,shape)
    gt = (gt/gt.max()*255).astype(np.int8)
    out = np.empty((conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH, 3), dtype=np.uint8)
    out[:, :, 0] = gt
    out[:, :, 1] = gt
    out[:, :, 2] = gt
    target_image = np.ubyte(0.8 * out + 0.2 * image)
    cv2.imwrite(name,target_image)

def test_make_label(image_path,json_path,charset):

    image= cv2.imread(image_path)
    f = open(json_path,encoding="utf-8")
    data = f.read()
    image_label = ImageLabel(image,data)

    generator = LabelGenerater(conf.MAX_SEQUENCE,
                               (conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH),
                                charset)
    character_segment,order_maps,localization_map = generator.process(image_label)

    if not os.path.exists(debug_dir):os.makedirs(debug_dir)

    save_image(os.path.join(debug_dir,"character_segment.jpg"),character_segment,image)
    save_image(os.path.join(debug_dir,"localization_map.jpg"), localization_map,image)
    for i,order_map in enumerate(order_maps):
        save_image(os.path.join(debug_dir,"order_map_{}.jpg".format(i+1)),order_map,image)

if __name__=="__main__":
    charset = label_utils.get_charset(charset_path)
    test_make_label("data/test/a.jpg","data/test/a.json",charset)