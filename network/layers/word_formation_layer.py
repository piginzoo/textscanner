import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf


class WordFormation(Layer):
    """
       integral the product of "Character Segmentation" & "Order Maps",
       and infer the character possibility.
       The threshold is 0.3(paper said "Other Detais: ... The score threshold L_score is set to 0.3 empirically...")
    """

    def __init__(self, name):
        super().__init__(name=name)

    def call(self, G, H, training=None):
        """
        G[Character Segmentation] : [N,H,W,C] - N:batch, C:charset size(3770)
        H[Order Map] :              [N,H,W,S] - S: Sequence Length(30)

        return will be [N,S,C],which means each character's probilities.
        """
        p_k_list = []
        for i in range(H.shape[-1]):
            H_k = H[:, :, :, i]
            H_k = H_k[:, :, :, tf.newaxis]
            GH = H_k * G
            p_k = K.sum(GH, axis=(1, 2))
            p_k_list.append(p_k)

        pks = K.stack(p_k_list)  # P_k: (30, 10, 4100)
        pks = K.permute_dimensions(pks, (1, 0, 2))
        return pks
