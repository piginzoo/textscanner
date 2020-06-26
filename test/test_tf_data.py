import tensorflow as tf
from utils.label import label_utils
from utils.label.image_label_loader import ImageLabelLoader


"""
测试如何如何使用tf.data，单独测试其使用
"""

charset = label_utils.get_charset("config/charset.txt")
# target_image_shape, charset, label_format, max_squence
iil = ImageLabelLoader((32,256),charset,"txt",30)

data_dir = "./data/train"

# [image_path, label_path]
data = label_utils.load_labels(data_dir)

path_ds = tf.data.Dataset.from_tensor_slices(data) # can be str

image_ds = path_ds.map(iil.load_one_image_label, num_parallel_calls=3)

# 提前加载额外的3个批次
image_ds = image_ds.prefetch(3)

it = image_ds.__iter__()
for i in range(10):
    x, y = it.next()
    print(x, y)

