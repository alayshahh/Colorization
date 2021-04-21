from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import random
from pprint import pprint
from cluster import Cluster
from constants import PATH_TO_GREYSCALE, PATH_TO_IMG, WIDTH, HEIGHT, RGB_VALUES, GREY_VALUES

ITERATIONS = 100

def dist(a, b):
    # we know that these are x,y,z tuples
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)



def kmeans() -> list:
    print(RGB_VALUES.shape)
    # Create initial cluster objects
    clusters = [
        Cluster([random.randint(0,256), random.randint(0,256), random.randint(0,256)]),
        Cluster([random.randint(0,256), random.randint(0,256), random.randint(0,256)]),
        Cluster([random.randint(0,256), random.randint(0,256), random.randint(0,256)]),
        Cluster([random.randint(0,256), random.randint(0,256), random.randint(0,256)]),
        Cluster([random.randint(0,256), random.randint(0,256), random.randint(0,256)])
    ]
    
    print("at the start the cluster values are:")
    for index, cluster in enumerate(clusters):
        print(f"\tcluster {index} is centered at {cluster.get_center()}")

    #define a delta that ensures that the k_means converges
    delta  = 1
    iteration = 0
    while delta>=0.05: #here we have defined convergence as the total change in for all cluster centers is less than 0.05
        delta = 0
        # we want to remove the points from every cluster since now we are recomputing which cluster they should
        # be belonging to
        for cluster in clusters:
            cluster.clear_points()

        for i in tqdm(range(HEIGHT), desc="pixel y axes"):
            for j in range(int(WIDTH/2)):
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
                candidate_cluster.add_point([RGB_VALUES[i,j, 0], RGB_VALUES[i, j, 1], RGB_VALUES[i, j, 2]])

        # now that we have grouped all of the clusters, we need to calculate a new center for all of the clusters
        # around each of their new points. We need to calculate the average point for each dimesnion (in our case: x, y, and z)
        for cluster in clusters:
            delta += cluster.update_center_around_points()

        # all the clusterse are
        print(f"at the end of iteration {iteration} the delta is: {delta} cluster values are:")
        iteration +=1
        for index, cluster in enumerate(clusters):
            print(f"\tcluster {index} is centered at {cluster.get_center()}, and the number of points in the cluster are {len(cluster.get_points())}")
    
    reduced_colors = np.zeros((HEIGHT, int(WIDTH/2), 3))
    for i in range(HEIGHT):
        for j in range(int(WIDTH/2)):
            candidate_cluster = clusters[0]
            distance_to_candidate_cluster = dist(
                    candidate_cluster.get_center(), RGB_VALUES[i, j])
            for cluster in clusters:
                    distance_to_new_cluster = dist(
                        cluster.get_center(), RGB_VALUES[i, j])
                    if distance_to_new_cluster < distance_to_candidate_cluster:
                        candidate_cluster = cluster
                        distance_to_candidate_cluster = distance_to_new_cluster
            center = candidate_cluster.get_center()
            reduced_colors[i,j,0], reduced_colors[i,j,1], reduced_colors[i,j,2] = center[0], center[1], center[2]

    print(RGB_VALUES.shape)
    print(reduced_colors.shape)
    image = Image.fromarray(reduced_colors.astype(np.uint8))
    image.save("./assets/reduced_colors.png")

    return clusters


# run kmeans
if __name__ == "__main__":
    clusters = kmeans()

    five_colors = np.array([[cluster.get_center() for cluster in clusters]]).astype(np.uint8)
    np.save("./assets/five_means_values.npy",five_colors)
    image = Image.fromarray(five_colors, 'RGB')
    image.save('./assets/kmeans_out.png')
    print("Saved image!")