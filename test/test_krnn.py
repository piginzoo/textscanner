from keras.layers import Layer
import keras.backend as K
from keras.layers import LSTM,Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class My_RNN(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim # 输出维度
        super(My_RNN, self).__init__(**kwargs)

    def build(self, input_shape): # 定义可训练参数
        self.kernel1 = self.add_weight(name='kernel1',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.kernel2 = self.add_weight(name='kernel2',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='kernel',
                                      shape=(self.output_dim,),
                                      initializer='glorot_normal',
                                      trainable=True)

    def step_do(self, step_in, states): # 定义每一步的迭代
        print("step_in:",step_in)
        print("states:",states)
        step_in  = tf.Print(step_in,[tf.shape(step_in)],"step_in")
        states  = tf.Print(states,[tf.shape(states)],"states")
        step_out = K.tanh(K.dot(states[0], self.kernel1) +
                          K.dot(step_in, self.kernel2) +
                          self.bias)
        return step_out, [step_out]

    def call(self, inputs): # 定义正式执行的函数
        init_states = [K.zeros((K.shape(inputs)[0],self.output_dim))] # 定义初始态(全零)
        print("init_states.shape:",init_states)
        outputs = K.rnn(self.step_do, inputs, init_states) # 循环执行step_do函数
        return outputs[0] # outputs是一个tuple，outputs[0]为最后时刻的输出，
                          # outputs[1]为整个输出的时间序列，output[2]是一个list，
                          # 是中间的隐藏状态。

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



train_X = np.random.rand(10,5,3)
train_y = np.random.rand(10,5)

model = Sequential()
model.add(My_RNN(output_dim=4, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(5))
model.compile(loss='mae', optimizer='adam')
model.summary()
# fit network
history = model.fit(train_X, train_y, epochs=2, batch_size=72,verbose=2, shuffle=False)
