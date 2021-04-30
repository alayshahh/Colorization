from improved_agent_functions import *
from PIL import Image
import numpy as np
from constants import *

if __name__ == '__main__':
    red_weights = start_training(Color.RED)
    print(red_weights)
    green_weights = start_training(Color.GREEN)
    print(green_weights)
    blue_weights = start_training(Color.BLUE)
    print(blue_weights)
    for i in range(HEIGHT-1):
        for j in range(WIDTH-1):
            x = get_x_vector((i, j))
            R = model(x, red_weights)*255
            G = model(x, green_weights)*255
            B = model(x, blue_weights)*255
            IMPROVED_OUT[i, j] = [R, G, B]
    print(IMPROVED_OUT)
    IMPROVED_OUT = IMPROVED_OUT.astype(np.uint8)
    np.save('./assets/improved_agent/out_matrix.npy', IMPROVED_OUT)
    img = Image.fromarray(IMPROVED_OUT)
    img.save('./assets/improved_agent/out.png')
