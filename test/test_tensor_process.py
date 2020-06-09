import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
G = np.random.random((10,64,256,512))
G = tf.convert_to_tensor(G)
H = np.random.random((10,64,256,3))
H = tf.convert_to_tensor(H)


p_k_list = []
for i in range(H.shape[-1]):
    H_k = H[:,:,:,i]
    H_k = H_k[:,:,:,tf.newaxis]
    print("H_k:",H_k.shape)
    GH = H_k*G
    print("GH:", GH.shape)
    p_k = K.sum(GH,axis=(1,2))
    print("p_k:", p_k.shape)
    print("------------")
    p_k_list.append(p_k)
pks = tf.stack(p_k_list) # P_k: (30, 10, 4100)
pks = K.permute_dimensions(pks, (1,0,2))
print("P_k:",pks.shape) # [10,30,4100]