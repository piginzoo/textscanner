import io
from tensorflow.keras.callbacks import Callback
from tensorflow.python.framework.ops import EagerTensor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import io,cv2
from PIL import Image

image = np.random.random((32,256))
buffer = io.BytesIO()
plt.imsave(buffer, image, format='jpg')
image = Image.open(buffer).convert('RGB')
image.save("../data/test.jpg")
image = np.array(image)
buffer.close()
image.shape

writer = tf.summary.create_file_writer("../data/tboard")
with writer.as_default():
    tf.summary.image("test123", np.array([image]), step=0)