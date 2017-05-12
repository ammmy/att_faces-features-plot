import numpy as np
from scipy.io import loadmat
feats_path = ["batch-%s.mat" % i for i in [0, 100, 200, 300]]
out_path = "face_feats"
feats = [loadmat(f)['feats'] for f in feats_path]
feats = np.concatenate(feats)
np.save(out_path, feats)

