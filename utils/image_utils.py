import cv2,numpy as np
import logging
import pyclipper
logger = logging.getLogger(__name__)


def show_image(img):
    # if img:plt.imshow(img)
    pass

# 图像缩放，高度都是64,这次的宽度，会和这个批次最宽的图像对齐填充padding
def read_and_resize_image(image_names: list,conf):

    padded_images = []

    for image_name in image_names:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("图像%s读取失败",image_name)
            continue
        # logger.debug("读取文件[%s]:%r",image_name,image.shape)
        h,w,_ = image.shape
        ratio = conf.INPUT_IMAGE_HEIGHT/h # INPUT_IMAGE_HEIGHT
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        show_image(image)
        # resize后，看实际宽度和要求的宽度，默认是256
        dim_difference = conf.INPUT_IMAGE_WIDTH - image.shape[1]
        if (dim_difference<0):
            # 如果图像宽了，就直接resize到最大
            padded_image = cv2.resize(image,(conf.INPUT_IMAGE_WIDTH,conf.INPUT_IMAGE_HEIGHT))
        else:
            # 否则，就给填充黑色,[(0, 0),(0, dim_difference),(0,0)]=>[高前后忽略,宽前忽略尾部加，通道前后忽略]
            padded_image = np.pad(image, [(0, 0),(0, dim_difference),(0,0)], 'constant',constant_values=(0))
        # show_image(padded_image)
        # cv2.imwrite("data/test.jpg", padded_image)
        padded_images.append(padded_image)
        # logger.debug("resize文件[%s]:%r", image_name, padded_image.shape)

    images = np.stack(padded_images,axis=0)
    # logger.debug("图像的shape：%r",images.shape)
    return images

def perimeter(polys):
    # 计算周长
    p = 0
    nums = polys.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(polys[i % nums] - polys[(i + 1) % nums]))
    # logger.debug('perimeter:{}'.format(p))
    return p

# 参考：https://blog.csdn.net/m_buddy/article/details/105614620
# polys[N,2]
def shrink_poly(polys, ratio=0.5):
    if type(polys)==list: polys = np.array(polys)
    """
    收缩多边形
    :param polys: 多边形
    :param ratio: 收缩比例
    :return:
    """
    area = abs(pyclipper.Area(polys)) # 面积
    _perimeter = perimeter(polys) # 周长

    pco = pyclipper.PyclipperOffset()
    if _perimeter:
        # TODO:不知道为何这样计算???
        d = area * (1 - ratio * ratio) / _perimeter
        pco.AddPath(polys, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 缩小后返回多边形
        polys_shrink = pco.Execute(-d)
    else:
        logger.warning("多边形周长为0")
        return None

    if len(polys_shrink)==0:
        logger.debug("收缩多边形[面积=%f]失败，使用原有坐标",area)
        return polys
    shrinked_bbox = np.array(polys_shrink[0])
    return shrinked_bbox

if __name__=="__main__":
    import conf
    read_and_resize_image("data/test.jpg", conf)
