from util import *
import matplotlib.pyplot as plt

# save g as image
def save_g_as_image(g, path):
    for i, _g in enumerate(g):
        img = to_rgb(np.array([_g])).reshape((h_raw, w_raw, rgb_channel_num))
        plt.imsave(path + "%s.jpg" % i, img)

# save g as binary
def save_g_as_binary(g, path):
    np.save(path + "g", g)

def save_g_as_image_and_binary(g, path):
    save_g_as_binary(g, path)
    save_g_as_image(g, path)

# Match rate between genes
def get_g_similarity(g):
    return [list(g[i] == g[i + 1]).count(True) / float(g.shape[1]) for i in range(g.shape[0] - 1)]

