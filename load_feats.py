# load feats of att_faces
# type: direct pixel, Vgg net, matlab SURF

import numpy as np
from scipy.io import loadmat
from util import *

def load_pixel_feats():
    return load_images()

def load_vgg_feats(feats_path="feats/Vgg16_Conv5_2/face_feats.npy"):
    return np.load(feats_path).reshape((npeople * nimage, -1))

def load_matlab_SURFFeats(feats_path='feats/SURFFeats.mat', mode='concat'):
    sf = loadmat(feats_path)['X'][0]
    h = max([x.shape[0] for x in sf])
    w = max([x.shape[1] for x in sf])
    sf_pad = np.zeros((sf.shape[0], h, w))
    for i, _sf in enumerate(sf):
        sf_pad[i][0:_sf.shape[0], 0:_sf.shape[1]] = _sf
    if mode == 'concat':
        sf_pad = sf_pad.reshape((sf.shape[0], -1))
    elif mode == 'sum':
        sf_pad = sf_pad.sum(1)
    return sf_pad

