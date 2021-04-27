from random import randrange
import numpy as np
from constants import *
import math
from basic_agent import get_patch
from vector import Vector
from enum import Enum


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
    values = np.array(values)
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
    L_i = -y_i ln(F(x_i)) - (1-y_i) ln(1-F(x_i))
    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return (-y*math.log(f) - (1-y)*math.log(1-f))


def update_weights(x: Vector, w: np.array, alpha: float, color: Color) -> np.array:
    '''
    returns the updated weight vector
    to update:  
        w_t+1 = w_t - alpha((F(x_i)-yi)*x_i)

    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return np.array(w - alpha*((f - y)*x.get_vector))


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
