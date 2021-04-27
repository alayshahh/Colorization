import blue_log_regression
import red_log_regression
import green_log_regression
from improved_agent_functions import *
from PIL import Image
import numpy as np
from constants import *


RED_TESTING = np.load('./assets/improved_agent/red_testing_loss.npy')
GREEN_TESTING = np.load('./assets/improved_agent/green_testing_loss.npy')
BLUE_TESTING = np.load('./assets/improved_agent/blue_testing_loss.npy')
RED_WEIGHTS = np.load('./assets/improved_agent/red_weights.npy')
GREEN_WEIGHTS = np.load('./assets/improved_agent/green_weights.npy')
BLUE_WEIGHTS = np.load('./assets/improved_agent/blue_weights.npy')

red_index = 0
min_val = float('inf')
for i, val in enumerate(RED_TESTING):
    if min_val > val:
        red_index = i
        min_val = val
RED_W = RED_WEIGHTS[red_index]
green_index = 0
min_val = float('inf')
for i, val in enumerate(GREEN_TESTING):
    if min_val > val:
        green_index = i
        min_val = val
GREEN_W = GREEN_WEIGHTS[green_index]
blue_index = 0
min_val = float('inf')
for i, val in enumerate(BLUE_TESTING):
    if min_val > val:
        blue_index = i
        min_val = val
BLUE_W = BLUE_WEIGHTS[blue_index]


for i in range(1, HEIGHT-1):
    for j in range(1, WIDTH-1):
        x = get_x_vector((i, j))
        R = model(x, RED_W)
        G = model(x, GREEN_W)
        B = model(x, BLUE_W)
        IMPROVED_OUT[i, j] = [R, G, B]
print(IMPROVED_OUT.shape)
IMPROVED_OUT = IMPROVED_OUT * 255
IMPROVED_OUT = IMPROVED_OUT.astype(np.uint8)
img = Image.fromarray(IMPROVED_OUT)
img.save('out.png')
