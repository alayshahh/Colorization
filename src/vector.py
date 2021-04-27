import numpy as np


class Vector:
    def __init__(self, location: tuple, values: np.array):
        self._i, self._j = location
        self._val = np.array(values)

    @property
    def get_location(self) -> tuple:
        '''
        Returns the location of the center pixel in the image
        '''
        return (self._i, self._j)

    @property
    def get_i(self) -> int:
        '''
        Returns the i/y location of the pixel in the image matrix
        '''
        return self._i

    @property
    def get_j(self) -> int:
        '''
        Returns the j/x location of the pixel in the image matrix
        '''
        return self._j

    @property
    def get_vector(self) -> np.array:
        '''
        Returns the x1,x2...x9 values in a numpy array
        '''
        return self._val
