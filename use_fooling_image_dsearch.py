# pred label with generated g

from config_GA import *
from util_GA import *
from util_dsearch import *
from GA import GA

model_name = "dsearch"
fd, m, sess = prepare_model()
feature_size = w_raw * h_raw

target_class = 0
ga = GA(feature_size, n_g=n_g, n_g_next=n_g_next, mut_rate=mut_rate, target_class=target_class)

ga.g = np.load("fooling_image/dsearch/g.npy")
ga.g = np.load("fooling_image/logreg/g.npy")

pred, cand = pred_nearest_neighbor(ga.g, m, sess)
accuracy = list(pred == ga.label).count(True) / float(ga.g.shape[0])

print cand, accuracy

