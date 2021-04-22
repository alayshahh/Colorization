from numpy import clongfloat
from constants import WIDTH, HEIGHT, GREY_VALUES, FIVE_MEANS_VALUES
import math
from queue import PriorityQueue
import numpy as np
from tqdm import tqdm


def get_patch(location: tuple) -> list:
    """
    Gets the 3x3 patch of pxiles with the given location tuple at the center
    """
    i, j = location
    return [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]


def find_delta(patch: list, cur_patch: list):
    """
    Calculates the difference between the
    """
    delta = 0
    for neighbor in range(9):
        x, y = patch[neighbor]
        black, white = GREY_VALUES[x, y, 0], GREY_VALUES[x, y, 1]
        i, j = cur_patch[neighbor]
        cur_black, cur_white = GREY_VALUES[i, j, 0], GREY_VALUES[i, j, 1]
        delta += (black-cur_black)**2 + (white - cur_white)**2
    return math.sqrt(delta)


def find_patches(location: tuple):
    """
    Finds the 6 closest patches in the training data that are similar to the current patch
    Returns a list of tuples containing the x,y coordinats of the center of the patch
    """
    closest_patches = []
    patch = get_patch(location)
    int(WIDTH/2 - 1)
    for i in range(1, HEIGHT-1):
        for j in range(1, int(WIDTH/2)-1):
            cur_patch = get_patch((i, j))
            # delta  = total difference in the B&W pxiel vlaues for current patch and original patch
            delta = find_delta(patch, cur_patch)
            if len(closest_patches) == 6:  # just need to find 6 closest patches
                # compare with greatest delta in closest patches
                min_delta, _ = closest_patches[5]
                if delta < min_delta:
                    closest_patches[5] = ((delta, (i, j)))
                    # sort based on delta
                    closest_patches.sort(key=lambda tup: tup[0])
            else:
                closest_patches.append((delta, (i, j)))
                closest_patches.sort(key=lambda tup: tup[0])
    print(closest_patches)
    return [closest_patches[i][1] for i in range(6)]


if __name__ == '__main__':
    similar_patches = []
    for x in tqdm(range(HEIGHT-1)):
        row_patches = []
        for y in range(int(WIDTH/2), int(WIDTH-1)):
            print(x, y)
            print(GREY_VALUES[x, y])
            patch = get_patch((x, y))
            row_patches.append(find_patches((x, y)))
        similar_patches.append(row_patches)
    similar_patches = np.array(similar_patches)
    print(similar_patches.shape)
    np.save('./assets/closest_patches.npy', similar_patches)
