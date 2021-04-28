import blue_log_regression
import red_log_regression
import green_log_regression
from improved_agent_functions import *
from PIL import Image
import numpy as np
from constants import *

if __name__ == '__main__':
    red_weights = red_log_regression.start()
    green_weights = green_log_regression.start()
    blue_weights = blue_log_regression.start()
    for i in range(HEIGHT-1):
        for j in range(WIDTH-1):
            x = get_x_vector((i, j))
            R = model(x, red_weights)
            G = model(x, green_weights)
            B = model(x, blue_weights)
            IMPROVED_OUT[i, j] = [R, G, B]
    IMPROVED_OUT = np.array(IMPROVED_OUT*255).astype(np.uint8)
    img = Image.fromarray(IMPROVED_OUT)
    img.save('./assets/improved_agent/out.png')
