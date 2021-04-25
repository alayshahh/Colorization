import k_means as k_means
from basic_agent import start
from constants import *
import numpy as np
from PIL import Image
from tqdm import tqdm


def loss(k: int) -> float:
    '''
    Computes the loss for a k_means clustering of the rgb values of the image
    Loss is the sum of the distance of actual rgb value compared to the k means color assigned to the pixel
    '''
    delta = 0
    _, reduced_colors = k_means.kmeans(k, 0.005)
    for i in range(HEIGHT):
        for j in range(int(WIDTH/2)):
            delta += k_means.dist(RGB_VALUES[i, j], reduced_colors[i, j])
    return delta


def find_best_k(k: int):
    '''
    Computes the loss of k means from 1 to param k
    used for graph in bonus_1.ipynb 
    '''
    losses = []
    for i in tqdm(range(1, k)):
        losses.append(loss(i))
    return losses


if __name__ == '__main__':

    # create the k means values for the inputted image and value of k
    colors, reduced_image = k_means.kmeans(k=11, alpha=0.005)

    # save the colors & the reduced colors array
    colors = np.array([[cluster.get_center()
                        for cluster in colors]]).astype(np.uint8)
    reduced_image = np.array(reduced_image)
    np.save('./assets/bonus_1/bonus_k_means.npy', colors.astype(np.uint8))
    np.save('./assets/bonus_1/k_colored.npy', reduced_image.astype(np.uint8))

    # create array for output of basic agent with 11 colors
    np.save('./assets/bonus_1/bonus_agent_1.npy',
            reduced_image.astype(np.uint8))
    colors = np.load('./assets/bonus_1/bonus_k_means.npy')
    reduced_image = np.load('./assets/bonus_1/k_colored.npy')
    out_matrix = np.array(np.load('./assets/bonus_1/bonus_agent_1.npy'))

    # save images
    img = Image.fromarray(colors)
    img.save('./assets/bonus_1/bonus_k_means.png')
    img = Image.fromarray(reduced_image)
    img.save('./assets/bonus_1/k_colored.png')

    # run basic agent with 11 colors and save image
    start(11, reduced_image, colors, out_matrix)
    np.save('./assets/bonus_1/bonus_agent_1.npy', out_matrix.astype(np.uint8))
    img = Image.fromarray(out_matrix)
    img.save('./assets/bonus_1/bonus_agent_1.png')
