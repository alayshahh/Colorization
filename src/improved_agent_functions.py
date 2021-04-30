from random import randrange
import numpy as np
from constants import *
import math
from basic_agent import get_patch
from vector import Vector
from enum import Enum
import random
from tqdm import tqdm


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def sigmoid(z: float) -> float:
    '''
    Appiles the sigmoid function to the given imput
        for logistic regression
    '''
    return math.pow(1+math.pow(math.e, -z), -1)


def get_x_vector(location: tuple) -> Vector:
    '''
    Based on the given location, a vector object is returned with the grey values of the patch in the Vectro and stores the location of the vector
    '''
    patch = get_patch(location)
    values = [1]
    for pixel in patch:
        i, j = pixel
        values.append(GREY_NORMALIZED[i, j])
    x = np.array(values).reshape(1, len(values))
    x_T = np.array(values).reshape(len(values), 1)
    values = np.matmul(x_T, x).flatten()
    return Vector(location, values)


def training_loss(w: np.array, color: Color) -> float:
    '''
    Gives total loss of model
    '''
    total_loss = 0
    for i in range(1, HEIGHT-1):
        for j in range(int(WIDTH/2)-1):
            x = get_x_vector((i, j))
            total_loss += li_loss(x, w, color)
    return total_loss


def li_loss(x: Vector, w: np.array, color: Color) -> float:
    '''
    Returns loss for one vector
    L_i = (F(x_i)- y_i)^2
    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return (f-y)**2


def update_weights(x: Vector, w: np.array, alpha: float, color: Color) -> np.array:
    '''
    returns the updated weight vector
    to update:  
        F(x) = sig(x.w)
        w_t+1 = w_t - alpha(2(F(x_i)-y_i)*F(x_i)(1-F(x_i)))x_i
    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return w - (alpha*(2*(f-y))*f*(1-f))*x.get_vector


def model(x: Vector, w: np.array) -> float:
    '''
    Returns sigmoid(x.w)
    '''
    return sigmoid(np.dot(w, x.get_vector))


def testing_loss(w: np.array, color: Color) -> float:
    '''
    Returns the loss for the testing data
    '''
    total_loss = 0
    for i in range(1, HEIGHT-1):
        for j in range(int(WIDTH/2)+1, WIDTH-1):
            x = get_x_vector((i, j))
            total_loss += li_loss(x, w, color)
    return total_loss


def start_training(color: Color) -> np.array:
    WEIGHTS = []
    TRAINING_LOSS = []
    TESTING_LOSS = []
    COLOR = Color.RED
    # initialize random weight vector (w @ t=0)
    w = np.array([random.uniform(-0.5, 0.5) for _ in range(100)])
    WEIGHTS.append(w)
    TRAINING_LOSS.append(training_loss(w, color))
    TESTING_LOSS.append(testing_loss(w, color))

    # stochiograd descent
    for iteration in tqdm(range(1000)):
        alpha = 10/(iteration+1)**(.5)
        # pick random x vector
        i = random.randint(0, HEIGHT-2)
        j = random.randint(0, int(WIDTH/2)-2)
        # print(i, j)
        x = get_x_vector((i, j))
        # update using SGD
        w = update_weights(x, w, alpha, color)
        WEIGHTS.append(w)
        TRAINING_LOSS.append(training_loss(w, color))
        TESTING_LOSS.append(testing_loss(w, color))
    if color == Color.RED:
        np.save('./assets/improved_agent/red_weights.npy', WEIGHTS)
        np.save('./assets/improved_agent/red_training_loss.npy', TRAINING_LOSS)
        np.save('./assets/improved_agent/red_testing_loss.npy', TESTING_LOSS)
    elif color == Color.GREEN:
        np.save('./assets/improved_agent/green_weights.npy', WEIGHTS)
        np.save('./assets/improved_agent/green_training_loss.npy', TRAINING_LOSS)
        np.save('./assets/improved_agent/green_testing_loss.npy', TESTING_LOSS)
    elif color == Color.BLUE:
        np.save('./assets/improved_agent/blue_weights.npy', WEIGHTS)
        np.save('./assets/improved_agent/blue_training_loss.npy', TRAINING_LOSS)
        np.save('./assets/improved_agent/blue_testing_loss.npy', TESTING_LOSS)

    min_index = 0
    min_val = float('inf')
    for i, val in enumerate(TESTING_LOSS):
        if min_val > val:
            min_index = i
            min_val = val
    return WEIGHTS[min_index]
