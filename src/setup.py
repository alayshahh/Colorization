from PIL import Image
import numpy as np
from constants import *
if __name__ == '__main__':
    img = Image.open(PATH_TO_IMG)
    rgb_array = np.asanyarray(img)
    grey_img = img.convert('LA')
    grey_img.save('./assets/greyscale_bird.png')
    bw_array = np.asanyarray(grey_img)
    np.save("./assets/rgb_values.npy", rgb_array)
    np.save("./assets/greyscale_values.npy", bw_array)
