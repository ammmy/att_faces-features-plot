# make fooling images against logreg model with GA
# TensorFlow is used for simple parallel calculation on GPU, not for differential

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

max_epoch = 5000
target_class = 0
ga = GA(feature_size, n_g=n_g, n_g_next=n_g_next, mut_rate=mut_rate, target_class=target_class)
for epoch in range(max_epoch):
    _fd = {m.x:ga.g, m.label:ga.label}
    scores, prob = sess.run([-m.cross_entropy_wo_softmax, m.prob], feed_dict=_fd)
    
    ga.step(scores)

    prob = sorted([p[target_class] for p in prob])[::-1]
    print prob[:10]
    if prob[10] == 1:break

# Match rate between genes
print get_g_similarity(ga.g)

# save g as image
save_g_as_image_and_binary(ga.g, "fooling_image/%s/" % model_name)

