# make sprite image for visualize features' thumbnails on TensorBoard

from util import *

images = load_images()
images_rgb = to_rgb(images)
images_rgb_reshaped = images_rgb.reshape((npeople * nimage, h_raw, w_raw, rgb_channel_num))
make_sprite_image(images_rgb_reshaped, w_thumb, h_thumb)

