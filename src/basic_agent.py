from constants import WIDTH, HEIGHT, GREY_VALUES, FIVE_COLORED_RGB_VALUES, FIVE_COLORS, BASIC_AGENT_VALUES, SIMILAR_PATCH
import math
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
        delta += (float(grey) - float(cur_grey))**2
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
    '''
    Computes the similar patches for the whole image
    run this once
    '''
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


def find_majority(location: tuple, max_color: list, K_MEANS_MATRIX, K_COLORS) -> list:
    '''
    Returns an array [r,g,b] for the majority color among the patch centers, 
    if no majority or tie, the color of the closest matching center is assigned
    '''
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
    if max < int(len(max_color)/2) + 1 or tie == True:  # if no majority or tie
        i, j = location
        # get the most similar patch center pixel
        x, y = int(SIMILAR_PATCH[i, j, 0, 0]), int(SIMILAR_PATCH[i, j, 0, 1])
        # return the RGB of the most similiar pixel
        return K_MEANS_MATRIX[x, y]
    else:
        # if there is a majority and no tie, then return the most prevalent color
        return K_COLORS[max_index]


def start(k: int, K_MEANS_MATRIX, K_COLORS, OUT_MATRIX):
    '''
    Color the right half of image based on the basic agent algorithm given
    '''
    for i in range(1, HEIGHT-1):
        for j in range(int(WIDTH/2)+1, WIDTH-1):
            # for each pixel in the testing data
            max_color = [0 for _ in range(k)]  # for each of the five colors
            for p in range(6):  # go through each of the 6 similar patches
                patch_i, patch_j = int(SIMILAR_PATCH[i,
                                                     j, p, 0]), int(SIMILAR_PATCH[i, j, p, 1])
                middle_pixel_color = K_MEANS_MATRIX[patch_i, patch_j]
                # increment max_color for the index of the middle pixel's color in FIVE_COLORS
                for color in range(5):
                    if middle_pixel_color[0] == K_COLORS[0, color, 0] and middle_pixel_color[1] == K_COLORS[0, color, 1] and middle_pixel_color[2] == K_COLORS[0, color, 2]:
                        max_color[color] += 1
                        break
            # go through to find majority color and set it in the new pixel
            OUT_MATRIX[i, j] = find_majority(
                (i, j), max_color, K_MEANS_MATRIX, K_COLORS)


if __name__ == '__main__':
    # get_similar_patches()  # run only once values saved in ./assets/closest_patches.npy
    start(5, FIVE_COLORED_RGB_VALUES, FIVE_COLORS, BASIC_AGENT_VALUES)
    np.save('./assets/basic_agent_values.npy', BASIC_AGENT_VALUES)
    img = Image.fromarray(BASIC_AGENT_VALUES)
    img.save('./assets/basic_agent.png')
