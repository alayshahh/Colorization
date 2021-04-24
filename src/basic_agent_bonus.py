import k_means as k_means
from basic_agent import start
from constants import *
import numpy as np
from PIL import Image
from tqdm import tqdm


def loss(k: int) -> float:
    delta = 0
    _, reduced_colors = k_means.kmeans(k, 0.005)
    for i in range(HEIGHT):
        for j in range(int(WIDTH/2)):
            delta += k_means.dist(RGB_VALUES[i, j], reduced_colors[i, j])
    return delta


def find_best_k(k: int):
    losses = []
    for i in tqdm(range(1, k)):
        losses.append(loss(i))
    return losses


if __name__ == '__main__':
    colors, reduced_image = k_means.kmeans(k=8, alpha=0.005)
    colors = np.array([[cluster.get_center()
                        for cluster in colors]]).astype(np.uint8)
    reduced_image = np.array(reduced_image)
    np.save('./assets/bonus_1/bonus_k_means.npy', colors.astype(np.uint8))
    np.save('./assets/bonus_1/k_colored.npy', reduced_image.astype(np.uint8))
    np.save('./assets/bonus_1/bonus_agent_1.npy',
            reduced_image.astype(np.uint8))
    colors = np.load('./assets/bonus_1/bonus_k_means.npy')
    reduced_image = np.load('./assets/bonus_1/k_colored.npy')
    out_matrix = np.array(np.load('./assets/bonus_1/bonus_agent_1.npy'))
    img = Image.fromarray(colors)
    img.save('./assets/bonus_1/bonus_k_means.png')
    img = Image.fromarray(reduced_image)
    img.save('./assets/bonus_1/k_colored.png')
    start(8, reduced_image, colors, out_matrix)
    np.save('./assets/bonus_1/bonus_agent_1.npy', out_matrix.astype(np.uint8))
    img = Image.fromarray(out_matrix)
    img.save('./assets/bonus_1/bonus_agent_1.png')
