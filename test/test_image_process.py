# 测试样本是否能被正确resize么？
from utils import image_utils
import conf
import matplotlib.pyplot as plt,cv2
#
# plt.title("processed images", fontsize='large', fontweight='bold')
#
# resize_images = image_utils.read_and_resize_image(["test/data/test/test1.jpg"], conf)
# plt.imshow(cv2.cvtColor(resize_images[0], cv2.COLOR_BGR2RGB))
# plt.show()
#
# resize_images = image_utils.read_and_resize_image(["test/data/test/test2.jpg"], conf)
# plt.imshow(cv2.cvtColor(resize_images[0], cv2.COLOR_BGR2RGB))
# plt.show()


# 测试收缩算法
import numpy as np
poly = np.array([[100,130],[140,126],[160,129],[170,140],[144,142],[124,135]])
shrinked_poly = image_utils.shrink_poly(poly,0.75)
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121)
ax.fill(poly[:,0],poly[:,1],'g')
ax = fig.add_subplot(121)
ax.fill(shrinked_poly[:,0],shrinked_poly[:,1],'r',alpha=0.8)
ax = fig.add_subplot(122)
ax.fill(shrinked_poly[:,0],shrinked_poly[:,1],'r',alpha=0.8)
plt.show()