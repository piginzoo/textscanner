import matplotlib.pyplot as plt
import numpy as np

csi = character_segment_image = np.random.random((64,256,3840))
csi = np.argmax(csi,axis=-1) # 需要把3840的那个维度，变成1

plt.imshow(csi)
plt.show()
