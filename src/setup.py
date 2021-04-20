from PIL import Image
import numpy as np
import math
from tqdm import tqdm

PATH_TO_IMG = "./assets/bird.png"
PATH_TO_GREYSCALE = "./assets/greyscale_bird.png"
WIDTH = 1125  # x/j
HEIGHT = 1123  # y/i
RGB_VALUES = np.load("./assets/rgb_values.npy")
GREY_VALUES = np.load("./assets/greyscale_values.npy")


def dist(a, b):
    # we know that these are x,y,z tuples
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


ITERATIONS = 10


class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = []

    def get_center(self):
        """
        Return the center point of the cluster as a [x,y,z] point
        """
        return self.center

    def get_points(self) -> list:
        """
        Return a list of [r,g,b] points
        """
        return self.points

    def add_point(self, point: list):
        """
        Add a [x,y,z] point to a cluser
        """
        self.points.append(point)

    def clear_points(self):
        """
        Remove all points from the list
        """
        self.points = []

    def update_center_around_points(self):
        """
        Recompute self.center to be the center of self.points
        """
        new_center = [0, 0, 0]  # x,y,z
        num_points = len(self.points)
        for dim in range(3):
            sum_over_dim = 0
            for point in self.points:
                sum_over_dim += point[dim]

            new_center[dim] = (1/num_points) * sum_over_dim

        self.center = new_center


def kmeans() -> list:
    # Create initial cluster objects
    clusters = [
        Cluster([51, 51, 51]),
        Cluster([102, 102, 102]),
        Cluster([153, 153, 153]),
        Cluster([204, 204, 204]),
        Cluster([255, 255, 255])
    ]

    for iteration in range(ITERATIONS):

        # we want to remove the points from every cluster since now we are recomputing which cluster they should
        # be belonging to
        for cluster in clusters:
            cluster.clear_points()

        for i in tqdm(range(HEIGHT), desc="pixel y axes"):
            for j in range(WIDTH):
                # candidate cluster is the cluster that the point is closest to
                candidate_cluster = clusters[0]
                distance_to_candidate_cluster = dist(
                    candidate_cluster.get_center(), RGB_VALUES[i, j])

                for cluster in clusters:
                    distance_to_new_cluster = dist(
                        cluster.get_center(), RGB_VALUES[i, j])
                    # if the distance to this cluster is smaller than the distance to the current
                    # candidate cluster, then this cluster is actually the best candidate cluster
                    # So, we should update our candidate cluster to to be this cluster (for this point)
                    if distance_to_new_cluster < distance_to_candidate_cluster:
                        candidate_cluster = cluster
                        distance_to_candidate_cluster = distance_to_new_cluster

                # add the point to the candidate cluster's list
                candidate_cluster.add_point(RGB_VALUES[i, j])

        # now that we have grouped all of the clusters, we need to calculate a new center for all of the clusters
        # around each of their new points. We need to calculate the average point for each dimesnion (in our case: x, y, and z)
        for cluster in clusters:
            cluster.update_center_around_points()

        # all the clusterse are
        print(f"at the end of iteration {iteration} the cluster values are:")
        for index, cluster in enumerate(clusters):
            print(f"\tcluster {index} is centered at {cluster.get_center()}")

    return clusters


# run kmeans
clusters = kmeans()

example_data = np.array([[cluster.get_center() for cluster in clusters]])
image = Image.fromarray(example_data, 'RGB')
image.save('kmeans_out.png')
print("Saved image!")
