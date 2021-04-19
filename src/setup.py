from PIL import Image
import numpy as np

PATH_TO_IMG = "./assets/bird.png"
PATH_TO_GREYSCALE = "./assets/greyscale_bird.png"
WIDTH = 1125 #x/j
LENGTH = 1123 #y/i
RGB_VALUES = np.load("./assets/rgb_values.npy")
GREY_VALUES = np.load("./assets/greyscale_values.npy")