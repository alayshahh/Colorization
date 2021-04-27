from constants import *
from improved_agent_functions import *
import numpy as np
import random
from tqdm import tqdm


def start() -> np.array:
    WEIGHTS = []
    TRAINING_LOSS = []
    TESTING_LOSS = []
    COLOR = Color.RED
    # initialize random weight vector (w @ t=0)
    w = np.array([random.uniform(-0.5, 0.5) for _ in range(10)])
    WEIGHTS.append(w)
    TRAINING_LOSS.append(training_loss(w, COLOR))
    TESTING_LOSS.append(testing_loss(w, COLOR))

    # stochiograd descent
    for iteration in tqdm(range(10000)):
        alpha = 10/(iteration+1)**(.5)
        # pick random x vector
        i = random.randint(0, HEIGHT-2)
        j = random.randint(0, int(WIDTH/2)-2)
        # print(i, j)
        x = get_x_vector((i, j))
        # update using SGD
        w = update_weights(x, w, alpha, COLOR)
        WEIGHTS.append(w)
        TRAINING_LOSS.append(training_loss(w, COLOR))
        TESTING_LOSS.append(testing_loss(w, COLOR))

    np.save('./assets/improved_agent/red_weights.npy', WEIGHTS)
    np.save('./assets/improved_agent/red_training_loss.npy', TRAINING_LOSS)
    np.save('./assets/improved_agent/red_testing_loss.npy', TESTING_LOSS)
    min_index = 0
    min_val = float('inf')
    for i, val in enumerate(TESTING_LOSS):
        if min_val > val:
            min_index = i
            min_val = val
    return WEIGHTS[min_index]


if __name__ == '__main__':
    print(start())
