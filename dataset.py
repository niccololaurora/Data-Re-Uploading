import numpy as np
from itertools import product


def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1
    return points, labels

def _tricrown(points):
    c = [[0, 0], [0, 0]]
    r = [np.sqrt(0.8), np.sqrt(0.8 - 2 / np.pi)]
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points - [c[0]], axis=1) > r[0])
    labels[ids] = 2
    ids = np.where(
        np.logical_and(
            np.linalg.norm(points - [c[0]], axis=1) < r[0],
            np.linalg.norm(points - [c[1]], axis=1) > r[1],
        )
    )
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

def create_target(name):
    """Function to create target states for classification.

    Args:
        name (str): Name of the problem to create the target states, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines']

    Returns:
        List of numpy arrays encoding target states that depend only on the number of classes of the given problem
    """
    if name in ["circle", "square", "crown"]:
        targets = [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
    elif name in ["tricrown"]:
        targets = [
            np.array([1, 0], dtype="complex"),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)], dtype="complex"),
            np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)], dtype="complex"),
        ]

    else:
        raise NotImplementedError("This dataset is not implemented")

    return targets

