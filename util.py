import matplotlib.pyplot as plt
from PIL import Image
import skimage
import skimage.transform
import matplotlib.cm as cm
import numpy as np
import os
from scipy.misc import imresize

from setting import *

def load_images():
    images = []
    file_name = './att_faces/s%s/%s.pgm'
    for i in range(npeople):
        for j in range(nimage):
            images.append(np.array(Image.open(file_name % (i + 1, j + 1)).getdata()))
    return np.cast[np.float32](images) / 255

def to_rgb(images):
    if len(images.shape) == 2:
        return images_2D_to_rgb(images)
    else:
        return images_3D_to_rgb(images)

def images_2D_to_rgb(images):
    return np.array([np.tile(im[:,np.newaxis] , (1,3)) for im in images])

def images_3D_to_rgb(images):
    return np.array([np.tile(im[:,:,np.newaxis] , (1,1,3)) for im in images])

def split_data(feats):
    split=["train", "dev", "test"]
    label = np.arange(npeople).repeat(nimage)
    idx = {}
    idx["all"] = np.arange(feats.shape[0])
    idx["dev"] = np.arange(npeople) * nimage
    idx["test"] = np.arange(npeople) * nimage + 1
    idx["train"] = np.array(list(set(idx["all"]) - set(idx["dev"]) - set(idx["test"])))
    return {k:feats[idx[k]] for k in split}, {k:label[idx[k]] for k in split}, {k:idx[k].shape[0] for k in split}

# tile images in a square, need to change
# see make_sprite_image.py
def make_sprite_image(images, w, h):
    images_thumb = [imresize(img, [h, w]) for img in images]
    images_thumb = np.cast[np.float32](images_thumb) / 255
    tile = [np.concatenate(images_thumb[i * nimage * 2:(i + 1) * nimage * 2], 1) for i in range(npeople / 2)]
    tile = np.concatenate(tile)
    plt.imsave(os.path.join(work_space, sprite_image_name), tile)

def make_metadata_file():
    metadata = np.cast[np.str]((np.arange(npeople) + 1).repeat(nimage))
    with open(os.path.join(work_space, metadata_name), "w") as f:
        f.write('\n')
        f.writelines('\n'.join(metadata))

