from random import randrange
from numpy import clongfloat
from numpy.lib.stride_tricks import sliding_window_view
from constants import WIDTH, HEIGHT, GREY_VALUES, FIVE_COLORED_RGB_VALUES, FIVE_COLORS, BASIC_AGENT_VALUES, SIMILAR_PATCH
import math
from queue import PriorityQueue
import numpy as np
from tqdm import tqdm
from PIL import Image


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
        grey = GREY_VALUES[x, y]
        i, j = cur_patch[neighbor]
        cur_grey = GREY_VALUES[i, j]
        delta += (grey - cur_grey)**2
    return math.sqrt(delta)


def find_patches(location: tuple):
    """
    Finds the 6 closest patches in the training data that are similar to the current patch
    Returns a list of tuples containing the x,y coordinats of the center of the patch
    """
    closest_patches = []
    patch = get_patch(location)
    for i in range(1, HEIGHT-1):  # 1-255
        for j in range(1, int(WIDTH/2) - 1):  # 1-127
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
    # print(closest_patches)
    return [closest_patches[i][1] for i in range(6)]


def get_similar_patches():
    # will hold the similar patches for each pixel in HEIGHT x WIDTH/2
    similar_patches = np.zeros((HEIGHT, WIDTH, 6, 2))
    # use (1,  HEIGHT -1) because we dont need to look at edge pixled b/c no 3x3 patch
    for x in tqdm(range(1, HEIGHT - 1)):  # 1-127
        for y in range(int(WIDTH/2) + 1, WIDTH - 1):  # 65 - 127
            # print(x, y)
            # GREY_VALUES[x, y]
            for i, patch in enumerate(find_patches((x, y))):
                similar_patches[x, y, i,
                                0], similar_patches[x, y, i, 1] = patch

    similar_patches = np.array(similar_patches)
    print(similar_patches.shape)
    np.save('./assets/closest_patches.npy', similar_patches)


def find_majority(location: tuple, max_color: list) -> list:
    max = 0
    max_index = 0
    tie = False
    for index, num in enumerate(max_color):
        if num > max:
            max = num
            max_index = index
            tie = False
        if num == max:
            tie = True
    if max < 3 or tie == True:  # if no majority or tie
        i, j = location
        # get the most similar patch center pixel
        x, y = int(SIMILAR_PATCH[i, j, 0, 0]), int(SIMILAR_PATCH[i, j, 0, 1])
        # return the RGB of the most similiar pixel
        return FIVE_COLORED_RGB_VALUES[x, y]
    else:
        # if there is a majority and no tie, then return the most prevalent color
        return FIVE_COLORS[max_index]


def start():
    for i in range(1, HEIGHT-1):
        for j in range(int(WIDTH/2)+1, WIDTH-1):
            # for each pixel in the testing data
            max_color = [0, 0, 0, 0, 0]  # for each of the five colors
            for p in range(6):  # go through each of the 6 similar patches
                patch_i, patch_j = int(SIMILAR_PATCH[i,
                                                     j, p, 0]), int(SIMILAR_PATCH[i, j, p, 1])
                middle_pixel_color = FIVE_COLORED_RGB_VALUES[patch_i, patch_j]
                # increment max_color for the index of the middle pixel's color in FIVE_COLORS
                for color in range(5):
                    if middle_pixel_color[0] == FIVE_COLORS[0, color, 0] and middle_pixel_color[1] == FIVE_COLORS[0, color, 1] and middle_pixel_color[2] == FIVE_COLORS[0, color, 2]:
                        max_color[color] += 1
                        break
            # go through to find majority color and set it in the new pixel
            BASIC_AGENT_VALUES[i, j] = find_majority((i, j), max_color)


if __name__ == '__main__':
    # TODO do the basic agent
    start()
    print(BASIC_AGENT_VALUES.shape)
    img = Image.fromarray(BASIC_AGENT_VALUES)
    img.save('./assets/basic_agent.png')
