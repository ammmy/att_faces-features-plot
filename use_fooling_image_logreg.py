# pred label with generated g

from config_logreg import *
from model_logreg import model
from util_tf import *
from config_GA import *
from util_GA import *
from GA import GA

def prepare_model(feature_size, pretrained_model_path):
    m, _ = build_model_logreg(model, class_num, feature_size, learning_rate)
    sess = make_and_init_sess(gpu_id="0")
    load_model(sess, pretrained_model_path)
    return m, sess

feature_size = w_raw * h_raw
pretrained_model_path="model/%s/" % model_name
m, sess = prepare_model(feature_size, pretrained_model_path)

target_class = 0
ga = GA(feature_size, n_g=n_g, n_g_next=n_g_next, mut_rate=mut_rate, target_class=target_class)

ga.g = np.load("fooling_image/dsearch/g.npy")
ga.g = np.load("fooling_image/logreg/g.npy")

target_class = 0
_fd = {m.x:ga.g, m.label:ga.label}
prob, pred = sess.run([m.prob, m.pred], feed_dict=_fd)
prob_target = sorted([p[target_class] for p in prob])[::-1]

print prob_target, pred

