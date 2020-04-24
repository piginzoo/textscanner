import numpy as np
import tensorflow as tf

def _p(t,name):
    print("调试计算图定义："+name, t)
    return tf.Print(t,[t],name,summarize=300)

# y_pred is [batch,seq,charset_size]
# pred  = np.random.rand(3,3,3)
# label = np.random.rand(3,3,3)

pred = np.array(
[
[[1,0,0],[1,0,0],[1,0,0]],
[[1,0,0],[1,0,0],[1,0,0]],
[[1,0,0],[1,0,0],[1,0,0]]
])

label = np.array(
[
[[0.5,0.2,0.3],[0.5,0.2,0.3],[0.5,0.2,0.3]], #true,true,true=>true
[[0.5,0.2,0.3],[0.5,0.2,0.3],[0.2,0.5,0.3]], #true,true,false=>false
[[0.2,0.3,0.5],[0.2,0.3,0.5],[0.2,0.3,0.5]]  #false,false,false=>false
])

# 正确率应该是0.333

def accuracy(y_true, y_pred):
    max_idx_p = tf.argmax(y_pred, axis=2)
    max_idx_l = tf.argmax(y_true, axis=2)
    max_idx_p = _p(max_idx_p,"max_idx_p")
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e), elems=correct_pred, dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

s = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, None,3], name='x')
y = tf.placeholder(tf.float32, shape=[None, None,3], name='y')
m = accuracy(x,y)
r = s.run(m,feed_dict={x:pred,y:label})
print(r)