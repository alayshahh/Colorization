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
    L_i = (y_i - F(x_i))^2
    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return (y - f)**2


def update_weights(x: Vector, w: np.array, alpha: float, color: Color) -> np.array:
    '''
    returns the updated weight vector
    to update:  
        F(x) = sig(x.w)
        w_t+1 = w_t - alpha(-2(y_i - F(x_i))*F(x_i)(1-F(x_i)))x_i
    '''
    i, j = x.get_location
    y = RGB_NORMALIZED[i, j, color.value]
    f = model(x, w)
    return w - (alpha*(-2*(y-f))*f*(1-f))*x.get_vector


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
