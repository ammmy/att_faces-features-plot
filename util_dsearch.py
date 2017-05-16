import numpy as np
from setting import *
import load_feats as lf
from model_dsearch import model
from util_tf import *
import collections

def split_data(feats):
    split = ["train", "test"]
    label = np.arange(npeople).repeat(nimage)
    idx = {k:None for k in split + ["all"]}
    idx["all"] = np.arange(feats.shape[0])
    idx["test"] = np.arange(npeople) * nimage
    idx["train"] = np.array(list(set(idx["all"]) - set(idx["test"])))
    return {s: {k:v for k, v in zip(["feats", "label"], [feats[idx[s]], label[idx[s]]])} for s in split}

def prepare_model(gpu_id="0"):
    feats = lf.load_pixel_feats()
    fd = split_data(feats)
    m = model(fd["train"]["feats"])
    sess = make_and_init_sess(gpu_id="0")
    return fd, m, sess

def calc_nearest_neighbor(m, g, sess, target_class, n_g):
    dist = sess.run(m.d, feed_dict={m.x:g})
    nearest_neighbor = np.argsort(dist) / (nimage - 1)
    idx = np.where(nearest_neighbor == target_class)[1]
    return idx.reshape((n_g, -1)).sum(1)

def pred_nearest_neighbor(g, m, sess, k=5):
    dist = sess.run(m.d, feed_dict={m.x:g})
    nearest_neighbor = dist.argsort()[:,:k] / (nimage - 1)
    count = [collections.Counter(_n) for _n in nearest_neighbor]
    return [_c.most_common()[0][0] for _c in count], count

