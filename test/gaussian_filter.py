#coding=utf-8
from PIL import Image
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi


def render_image(): 
     im = np.array(Image.open('messi.jpg'))
      
     index = 141  #画1行四列的图，与 1,4,1 同
     plt.subplot(index)
     plt.imshow(im)
      
     for sigma in (2, 5, 10):
         im_blur = np.zeros(im.shape, dtype=np.uint8)
         for i in range(3):  #对图像的每一个通道都应用高斯滤波
             im_blur[:,:,i] = filters.gaussian_filter(im[:,:,i], sigma)
         index += 1
         plt.subplot(index)
         plt.imshow(im_blur)
      
     plt.show()


def render_gaussian(h,w,box):
    canvas = np.zeros((h,w), dtype=np.int32)
    xmin, xmax,ymin, ymax = box
    out = np.zeros_like(canvas).astype(np.float32)
    h, w = canvas.shape[:2]
    sigma = 2

    # 求中心点
    y = (ymax+ymin+1)//2
    x = (xmax+xmin+1)//2

    # 那个点上值为1    
    out[y, x] = 1.
    print("============================================================")
    print("原始out")
    print(out)
    # 
    h, w = canvas.shape[:2]
    fi.gaussian_filter(out, (sigma, sigma),output=out, mode='mirror')
    
    print("============================================================")
    print("高斯过滤后out")
    print(out)
    plt.subplot(131)#画1行四列的图，与 1,4,1 同
    plt.imshow(out)

    out = out / out.max()
    print("============================================================")
    print("归一化后out")
    print(out)    
    plt.subplot(132)#画1行四列的图，与 1,4,1 同
    plt.imshow(canvas)

    canvas[out > canvas] = out[out > canvas]
    print("============================================================")
    print("重新填充后的canvas")
    print(out)    
    plt.subplot(133)#画1行四列的图，与 1,4,1 同
    plt.imshow(canvas)
    
    plt.show()


def render_gaussian_thresh(h,w,box):
    canvas = np.zeros((h,w), dtype=np.int32)
    xmin, xmax,ymin, ymax = box
    value=7
    thresh=0.2
    shrink=0.6
    sigma = 2
    out = np.zeros_like(canvas)
    h, w = canvas.shape[:2]
    y = (ymax+ymin+1)//2
    x = (xmax+xmin+1)//2

    out = np.zeros_like(canvas).astype(np.float32)
    print(out.shape)
    out[y, x] = 1.
    print("============================================================")
    print("原始out")
    print(out)

    # out = filters.gaussian_filter(out,sigma=3)
    fi.gaussian_filter(out, (sigma, sigma),output=out, mode='mirror')
    print("============================================================")
    print("高斯滤波后out")
    print(out)
    out = out / out.max()
    print("============================================================")
    print("归一化后out")
    print(out)
    canvas[out > thresh] = value
    print("============================================================")
    print("out大于0.2复制了%d的canvas" % value)
    print(canvas)
    plt.imshow(canvas)
    plt.show()

if __name__ == '__main__':
    #render_gaussian_thresh(64,256,(30,50,30,50))
    render_gaussian(8,8,(3,5,3,5))