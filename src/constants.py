import numpy as np

PATH_TO_IMG = "./assets/bird.png"
PATH_TO_GREYSCALE = "./assets/greyscale_bird.png"
WIDTH = 128  # x/j
HEIGHT = 128  # y/i
RGB_VALUES = np.load("./assets/rgb_values.npy")
GREY_VALUES = np.load("./assets/greyscale_values.npy")
FIVE_COLORED_RGB_VALUES = np.load('./assets/five_colored.npy')
FIVE_COLORS = np.load('./assets/five_means.npy')
SIMILAR_PATCH = np.load('./assets/closest_patches.npy')
BASIC_AGENT_VALUES = np.load('./assets/basic_agent_values.npy')
