# make fooling images against dsearch model with GA
# TensorFlow is used for simple parallel calculation on GPU, not for differential

from config_GA import *
from util_GA import *
from util_dsearch import *
from GA import GA

model_name = "dsearch"
fd, m, sess = prepare_model()
feature_size = w_raw * h_raw

max_epoch = 5000
target_class = 0
ga = GA(feature_size, n_g=n_g, n_g_next=n_g_next, mut_rate=mut_rate, target_class=target_class)

for epoch in range(max_epoch):
    scores = -calc_nearest_neighbor(m, ga.g, sess, ga.target_class, ga.n_g)

    ga.step(scores)

    scores.sort()
    accuracy = list(pred_nearest_neighbor(ga.g, m, sess)[0] == ga.label).count(True) / float(ga.g.shape[0])
    print accuracy, 
    print scores[-10:][::-1]
    if accuracy == 1:break

# accuracy
list(pred_nearest_neighbor(ga.g, m, sess)[0] == ga.label).count(True) / float(ga.g.shape[0])

# Match rate between genes
print get_g_similarity(ga.g)

# save g as image
save_g_as_image_and_binary(ga.g, "fooling_image/%s/" % model_name)

