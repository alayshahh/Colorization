
from PIL import Image
import numpy as np
from constants import *


def enlarge_rgb_for_report(image: list, factor: int = 1):
    """
    Given a grid that came from generate_grid(), return
    a PIL Image object that can be either saved as a file.

    factor will scale the image by a factor of len(grid)
    agent is a tuple (i,j) representing the i,j position of the agent (it will draw a dot there on the image)
    """

    dim1, dim2, dim3 = image.shape

    image_array = np.zeros(
        (dim1 * factor, dim2 * factor, dim3), dtype=np.uint8)
    for i in range(dim1):
        for j in range(dim2):
            r, g, b = image[i, j, 0], image[i,
                                            j, 1], image[i, j, 2]

            for ki in range(factor):
                for kj in range(factor):
                    image_array[i * factor + ki, j *
                                factor + kj, 0] = r  # red channel
                    image_array[i * factor + ki, j * factor + kj,
                                1] = g  # green channel
                    image_array[i * factor + ki, j * factor + kj,
                                2] = b  # blue channel
    print(image_array.shape)
    img = Image.fromarray(image_array)
    return img


def enlarge_greyscale_for_report(image: list, factor: int = 1):
    """
    Given a grid that came from generate_grid(), return
    a PIL Image object that can be either saved as a file.

    factor will scale the image by a factor of len(grid)
    agent is a tuple (i,j) representing the i,j position of the agent (it will draw a dot there on the image)
    """

    dim1, dim2 = image.shape

    image_array = np.zeros(
        (dim1 * factor, dim2 * factor), dtype=np.uint8)
    for i in range(dim1):
        for j in range(dim2):
            bw = image[i, j]
            for ki in range(factor):
                for kj in range(factor):
                    image_array[i * factor + ki, j *
                                factor + kj] = bw  # red channel
    print(image_array.shape)
    img = Image.fromarray(image_array, 'L')
    return img


if __name__ == '__main__':
    enlarge_rgb_for_report(RGB_VALUES, 25).save(
        './assets/report/enlarged_bird.png')
    enlarge_greyscale_for_report(GREY_VALUES, 25).save(
        './assets/report/enlarged_greyscale_bird.png')
    enlarge_rgb_for_report(BASIC_AGENT_VALUES, 25).save(
        './assets/report/enlarged_basic_agent.png')
    enlarge_rgb_for_report(FIVE_COLORS, 50).save(
        './assets/report/enlarged_five_means_out.png')
    enlarge_rgb_for_report(FIVE_COLORED_RGB_VALUES, 25).save(
        './assets/report/enlarged_five_coloerd.png')
