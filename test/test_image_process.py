# 测试样本是否能被正确resize么？
from utils import image_utils
from main import conf
import matplotlib.pyplot as plt,cv2

plt.title("processed images", fontsize='large', fontweight='bold')

resize_images = image_utils.read_and_resize_image(["test/data/test1.jpg"],conf)
plt.imshow(cv2.cvtColor(resize_images[0], cv2.COLOR_BGR2RGB))
plt.show()

resize_images = image_utils.read_and_resize_image(["test/data/test2.jpg"],conf)
plt.imshow(cv2.cvtColor(resize_images[0], cv2.COLOR_BGR2RGB))
plt.show()
