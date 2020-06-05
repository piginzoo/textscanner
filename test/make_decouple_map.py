import numpy as np
import scipy.ndimage.filters as fi
import ipdb
from concern.config import State

from .data_process import DataProcess


class MakeDecoupleMap(DataProcess):
    max_size = State(default=32) # ？？？什么size，是最多的字符数么？
    shape = State(default=(64, 256)) # 图像的标准宽度
    sigma = State(default=2) # 噢，方差设成了2
    summation = State(default=False) 
    box_key = State(default='charboxes') 
    function = State(default='gaussian')
    thresh = State(default=0.2)
    order_dest = State(default='ordermaps')
    mask_dest = State(default='charmaps')
    shape_dest = State(default='shapemaps')

    def process(self, data):
        assert self.box_key in data, '%s in data is required' % self.box_key
        shape = data['image'].shape[:2] # h,w
        boxes = np.array(data[self.box_key]) # 这个是单字的框

        ratio_x = shape[1] / self.shape[1] # 高度比
        boxes[:, :, 0] = (boxes[:, :, 0] / ratio_x).clip(0, self.shape[1]) # 估计是boxes是所有的框:[b,N,2]
        ratio_y = shape[0] / self.shape[0] # 宽度比
        boxes[:, :, 1] = (boxes[:, :, 1] / ratio_y).clip(0, self.shape[0])
        boxes = (boxes + .5).astype(np.int32)
        xmins = boxes[:, :, 0].min(axis=1) # 找到x最小的值
        xmaxs = np.maximum(boxes[:, :, 0].max(axis=1), xmins + 1) # 找到x最大值
        ymins = boxes[:, :, 1].min(axis=1)
        ymaxs = np.maximum(boxes[:, :, 1].max(axis=1), ymins + 1)

        # 做了一张空图，h,w，全0
        shapemaps = np.zeros((self.shape[0], self.shape[1], 2), dtype=np.int32)


        if self.summation: 
            # 感觉是给localization map准备的gt
            canvas = np.zeros(self.shape, dtype=np.int32)
        else:
            # 3维度的，有点像是 order map的gt
            canvas = np.zeros((self.max_size+1, *self.shape), dtype=np.float32)

        mask = np.zeros(self.shape, dtype=np.float32)
        # 生成1~30的序号
        orders = self.orders(data)

        # 处理每个字符       
        for i in range(xmins.shape[0]):
            # 初始化一个h,w的零图
            temp = np.zeros(self.shape, dtype=np.float32)
            function = getattr(self, 'render_' + self.function)
            order = min(orders[i], self.max_size)
            if self.summation:
                function(canvas, xmins[i], xmaxs[i], ymins[i], ymaxs[i],
                         value=order+1, shrink=0.6)
            else:
                # 这个是每张图
                function(canvas[order+1], xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            self.render_gaussian(mask, xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            self.render_gaussian(temp, xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            w, h = xmaxs[i]-xmins[i], ymaxs[i]-ymins[i]
            shapemaps[temp > 0.4] = np.array([w, h])
        data[self.order_dest] = canvas
        data[self.mask_dest] = mask
        data[self.shape_dest] = shapemaps.transpose(2, 0, 1)
        return data

    def render_gaussian(self, canvas, xmin, xmax, ymin, ymax):
        out = np.zeros_like(canvas)
        h, w = canvas.shape[:2]
        # 求中心点
        y = (ymax+ymin+1)//2
        x = (xmax+xmin+1)//2
        if not (w > x \\and h > y): return
        # 那个点上值为1    
        out[y, x] = 1.
        h, w = canvas.shape[:2]
        fi.gaussian_filter(out, (self.sigma, self.sigma),output=out, mode='mirror')
        out = out / out.max()
        canvas[out > canvas] = out[out > canvas]# <--- 

    def render_gaussian_thresh(self, canvas, xmin, xmax, ymin, ymax,
                               value=1, thresh=None, shrink=None):
        if thresh is None:thresh = self.thresh
        h, w = canvas.shape[:2]
        y = (ymax+ymin+1)//2
        x = (xmax+xmin+1)//2
        if not (w > x and h > y):return
        out = np.zeros_like(canvas).astype(np.float32)
        out[y, x] = 1.
        out = fi.gaussian_filter(out, (self.sigma, self.sigma),output=out, mode='mirror')
        out = out / out.max()
        canvas[out > thresh] = value


    def render_gaussian_fast(self, canvas, xmin, xmax, ymin, ymax):
        out = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.float32)
        out[(ymax-ymin+1)//2, (xmax-xmin+1)//2] = 1.
        h, w = canvas.shape[:2]
        fi.gaussian_filter(out, (self.sigma, self.sigma),
                           output=out, mode='mirror')
        out = out / out.max()
        canvas[ymin:ymax+1, xmin:xmax+1] = np.maximum(out, canvas[ymin:ymax+1, xmin:xmax+1])

    def orders(self, data):
        orders = []
        if 'lines' in data: # lines什么鬼？
            for text in data['lines'].texts:
                orders += list(range(min(len(text), self.max_size)))
        else:
            # 我理解，就是生成了一个1：max_size的序号（data[self.box_key]是就是box们）
            orders = list(range(min(data[self.box_key].shape[0], self.max_size)))
        return orders