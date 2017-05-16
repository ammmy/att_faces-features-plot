# simple logreg model

import tensorflow as tf

class model():
    def __init__(self, class_num, feature_size):
        self.class_num, self.feature_size = class_num, feature_size
        self.W = tf.Variable(tf.random_uniform([self.feature_size, self.class_num], -1.0, 1.0), name='W')
        self.b = tf.Variable(tf.random_uniform([self.class_num], -1.0, 1.0), name='b')

    def build_model(self):
        x = tf.placeholder(tf.float32, [None, self.feature_size])
        label = tf.placeholder(tf.int64, [None])

        with tf.variable_scope("prediction"):
            logit = tf.matmul(x, self.W) + self.b
            cross_entropy_wo_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label)
            pred = tf.argmax(logit, 1)
            prob = tf.nn.softmax(logit)
        with tf.variable_scope("training"):
            logit = tf.matmul(tf.nn.dropout(x, 0.5), self.W) + self.b
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label)
            loss = tf.reduce_sum(cross_entropy) + tf.nn.l2_loss(self.W)

        with tf.variable_scope("metrics"):
            self.accuracy = tf.contrib.metrics.accuracy(pred, label)

        self.x, self.label = x, label
        self.logit, self.cross_entropy, self.pred, self.loss, self.prob = logit, cross_entropy, pred, loss, prob
        self.cross_entropy_wo_softmax = cross_entropy_wo_softmax

