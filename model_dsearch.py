# simple dsearch model

import tensorflow as tf

class model():
    def __init__(self, feats):
        p = tf.constant(feats)
        x = tf.placeholder(tf.float32, [None, feats.shape[1]])
        d = tf.pow(tf.subtract(p, tf.expand_dims(x, 1)), 2)
        d = tf.pow(tf.reduce_sum(d, 2), 0.5)
        self.p, self.x, self.d = p, x, d

