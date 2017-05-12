# tensorboard --logdir=LOG_DIR for TensorBoard

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from config import *
import os
import load_feats as lf

def make_session(gpu_id):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id
    sess = tf.Session(config=config)
    return sess

def build(feats):
    embedding_var = tf.Variable(feats.reshape((npeople * nimage, -1)), name='image_feature')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = metadata_name
    embedding.sprite.image_path = sprite_image_name
    embedding.sprite.single_image_dim.extend([w_thumb, h_thumb])

    summary_writer = tf.summary.FileWriter(work_space)
    projector.visualize_embeddings(summary_writer, config)

def write_embedding(feats):
    build(feats)
    sess = make_session(gpu_id='0')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(work_space, "model.ckpt"), global_step=0)

# sample
f = [lf.load_pixel_feats, lf.load_vgg_feats, lf.load_matlab_SURFFeats]
feats = f[0]()
feats = f[1](feats_path="feats/Vgg16_Conv5_2/face_feats.npy")
feats = f[1](feats_path="feats/Vgg16_fc7/face_feats.npy")
feats = f[2](mode='concat')
feats = f[2](mode='sum')
write_embedding(feats)

