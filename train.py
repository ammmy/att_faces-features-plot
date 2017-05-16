import load_feats as lf
from config_logreg import *
from model_logreg import model
from util import *
from util_tf import *

# feats type selection
feats = lf.load_pixel_feats()
feats = lf.load_vgg_feats(feats_path="feats/Vgg16_Conv5_2/face_feats.npy")
feats = lf.load_vgg_feats(feats_path="feats/Vgg16_fc7/face_feats.npy")
feats = lf.load_matlab_SURFFeats(mode="concat")
feats = lf.load_matlab_SURFFeats(mode="sum")

feature_size = feats.shape[1]
m, train_op = build_model_logreg(model, class_num, feature_size, learning_rate)
sess = make_and_init_sess(gpu_id="0")

feats_split, label_split, data_size = split_data(feats)
fd = make_feeder_logreg(feats_split, label_split, [m.x, m.label])

batch_size, train_data_size = data_size["train"], data_size["train"]
start_epoch, max_epoch = 0, 5000
for epoch in range(start_epoch, max_epoch):
    for kk, start in enumerate(range(0, train_data_size, batch_size)):
        end = start + batch_size
        if end > train_data_size:break
        _, train_loss = sess.run([train_op, m.loss], feed_dict=fd["train"].get(start, end, rnd=True))

    train_loss, train_accuracy = sess.run([m.loss, m.accuracy], feed_dict=fd["train"].get())
    dev_loss, dev_accuracy = sess.run([m.loss, m.accuracy], feed_dict=fd["dev"].get())
    print "@epoch %s, train_loss : %s, dev_loss : %s, train_accuracy : %s, dev_accuracy : %s" % (epoch, train_loss, dev_loss, round(train_accuracy, 4), round(dev_accuracy, 4))

saver = tf.train.Saver()
saver.save(sess, "model/%s/model.ckpt" % model_name, global_step=epoch)

