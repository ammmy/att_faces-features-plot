import tensorflow as tf
from feeder import feeder

def make_sess(gpu_id="0"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id
    sess = tf.Session(config=config)
    return sess

def init_sess(sess):
    sess.run(tf.global_variables_initializer())

def make_and_init_sess(gpu_id="0"):
    sess = make_sess(gpu_id)
    init_sess(sess)
    return sess

def build_model_logreg(model, class_num, feature_size, learning_rate):
    m = model(class_num, feature_size)
    m.build_model()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(m.loss)
    return m, train_op

def make_feeder_logreg(feats, label, keys, split=["train", "dev", "test"]):
    return {k:feeder(keys, [feats[k], label[k]]) for k in split}

def load_model(sess, pretrained_model_path):
    ckpt = tf.train.get_checkpoint_state(pretrained_model_path)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

