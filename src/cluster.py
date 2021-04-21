import random

from PIL.Image import blend

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

    def update_center_around_points(self) -> float:
        """
        Recompute self.center to be the center of self.points
        Returns the total change that the update caused, used for tracking convergence
        """
        old_center = self.center
        new_center = [0, 0, 0]  # x,y,z
        num_points = len(self.points)
        if num_points == 0:
            new_center = [random.randint(0,256), random.randint(0,256), random.randint(0,256)]
        else:
            red_val, green_val, blue_val = 0,0,0
            for point in self.points:
                red_val += point[0]
                green_val += point[1]
                blue_val += point[2]
            red_val = red_val/num_points
            green_val = green_val/num_points
            blue_val = blue_val/num_points
            new_center = [red_val, green_val, blue_val]
            

        self.center = new_center
        return abs(new_center[0]- old_center[0]) + abs(new_center[1]- old_center[1]) + abs(new_center[2]- old_center[2])
