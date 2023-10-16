import numpy as np
from itertools import product


def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1
    return points, labels

def create_dataset(grid=None, samples=1000, seed=0):
    """Function to create training and test sets for classifying.

    Args:
        samples (int): Number of points in the set, randomly located.
        seed (int): Random seed

    Returns:
        Dataset for the given problem (x, y)
    """
    if grid == None:
        print("Grid none")
        np.random.seed(seed)
        points = 1 - 2 * np.random.rand(samples, 2)
        print(f"Length Points {len(points)}")
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    creator = globals()["_circle"]

    x, y = creator(points)

    return x, y


