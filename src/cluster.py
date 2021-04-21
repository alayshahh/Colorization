import random

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
        if num_points == 0:
            new_center = [random.randint(0,256), random.randint(0,256), random.randint(0,256)]
        else:
            for dim in range(3):
                sum_over_dim = 0
                for point in self.points:
                   sum_over_dim += point[dim]

                new_center[dim] = (1/num_points) * sum_over_dim

        self.center = new_center
