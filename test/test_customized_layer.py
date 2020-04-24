# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import LeakyReLU
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.backend import squeeze
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Layer,Flatten,Dense
from keras.backend import squeeze
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
import numpy as np

class Conv(Layer):

    #[N,1,256/4,512] => [N,256/4,512]
    def squeeze_wrapper(self,tensor):
        print("tensor:",tensor)
        return squeeze(tensor, axis=1)

    def __init__(self, **kwargs):
        super(Conv, self).__init__(**kwargs)

    '''
        #抽feature，用的cnn网络
        # https://blog.csdn.net/Quincuntial/article/details/77679463
        在CRNN模型中，通过采用标准CNN模型（去除全连接层）中的卷积层和最大池化层来构造卷积层的组件。
        这样的组件用于从输入图像中提取序列特征表示。在进入网络之前，所有的图像需要缩放到相同的高度。
        然后从卷积层组件产生的特征图中提取特征向量序列，这些特征向量序列作为循环层的输入。
        具体地，特征序列的每一个特征向量在特征图上按列从左到右生成。这意味着第i个特征向量是所有特征图第i列的连接。
        在我们的设置中每列的宽度固定为单个像素。

        # 由于卷积层，最大池化层和元素激活函数在局部区域上执行，因此它们是平移不变的。
        因此，特征图的每列对应于原始图像的一个矩形区域（称为感受野），并且这些矩形区域与特征图上从左到右的相应列具有相同的顺序。
        如图2所示，特征序列中的每个向量关联一个感受野，并且可以被认为是该区域的图像描述符。
        :param inputdata: eg. batch*32*100*3 NHWC format
          |
        Conv1  -->  H*W*64          #卷积后，得到的维度
        Relu1
        Pool1       H/2 * W/2 * 64  #池化后得到的维度
          |
        Conv2       H/2 * W/2 * 128
        Relu2
        Pool2       H/4 * W/4 * 128
          |
        Conv3       H/4 * W/4 * 256
        Relu3
          |
        Conv4       H/4 * W/4 * 256
        Relu4
        Pool4       H/8 * W/4 * 64
          |
        Conv5       H/8 * W/4 * 512
        Relu5
        BatchNormal5
          |
        Conv6       H/8 * W/4 * 512
        Relu6
        BatchNormal6
        Pool6       H/16 * W/4 * 512
          |
        Conv7
        Relu7       H/32 * W/4 * 512
          |
          20层
    '''
    # 自定义的卷基层，32x100 => 1 x 25，即（1/32，1/4)
    def call(self,inputs):
        x = inputs
        for layer in self.layers:
            # print(x)
            x = layer(x)

        return x


    def build(self, input_shape):
        self.layers = []
        # Block 1
        self.layers.append(Conv2D(64, (3, 3), padding='same', name='block1_conv1'))
        self.layers.append(LeakyReLU())
        # self.layers.append(BatchNormalization())
        self.layers.append(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')) #1/2

        # Block 2
        self.layers.append(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
        self.layers.append(LeakyReLU())
        # self.layers.append(BatchNormalization())
        self.layers.append(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')) #1/2

        # Block 3
        self.layers.append(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
        self.layers.append(LeakyReLU())
        # self.layers.append(BatchNormalization())

        # Block 4
        self.layers.append(Conv2D(256, (3, 3), padding='same', name='block4_conv1'))
        # self.layers.append(BatchNormalization())
        self.layers.append(LeakyReLU())
        self.layers.append(MaxPooling2D((2, 1), strides=(2, 1), name='block4_pool')) # 1/2 <------ pool kernel is (2,1)!!!!!

        # Block 5
        self.layers.append(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
        self.layers.append(LeakyReLU())
        self.layers.append(BatchNormalization())

        # Block 6
        self.layers.append(Conv2D(512, (3, 3), padding='same', name='block6_conv1'))
        self.layers.append(LeakyReLU())
        self.layers.append(BatchNormalization())
        self.layers.append(MaxPooling2D((2, 1), strides=(2, 1), name='block6_pool')) #1/2 <------ pool kernel is (2,1)!!!!!

        # Block 7
        self.layers.append(Conv2D(512, (2, 2), strides=[2, 1], padding='same', name='block7_conv1')) #1/2
        self.layers.append(LeakyReLU())

        # 输出是(batch,1,Width/4,512),squeeze后，变成了(batch,Width/4,512)
        self.layers.append(Lambda(self.squeeze_wrapper))

        super(Conv, self).build(input_shape)

    # # input_shape[N,H,W,512] => output_shape[N,W/4,512]
    def compute_output_shape(self, input_shape):
        print("input_shape:",input_shape)
        return (None, int(input_shape[2]/4),512)

if __name__ == '__main__':

    input_image = Input(shape=(32,256,3))
    conv = Conv()
    conv_output = conv(input_image) # output[64,512]
    print(conv_output)
    flat = Flatten()(conv_output)
    output = Dense(4,activation='softmax',input_shape=(-1,))(flat)

    train_model = Model(inputs=input_image, outputs=output)
    adam = Adam()
    train_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    train_data = np.random.random((10,32,256,3))
    train_labels = np.random.random((10,4))
    train_model.fit(train_data, train_labels, epochs=1, batch_size=1)
