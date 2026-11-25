import numpy as np

def save_by_distance(panel, distances, metrics):
    """Pool node observations across realizations, using population standard deviations."""
    distance = np.concatenate([np.ravel(row) for row in distances])
    bins = np.unique(distance)
    for name, trials in metrics.items():
        values = np.concatenate([np.ravel(row) for row in trials])
        if values.shape != distance.shape:
            raise ValueError(f"{panel}/{name}: distance and observation counts differ")
        groups = [values[distance == d] for d in bins]
        np.save(f"scaling/{panel}_{name}_ER.npy",
                [bins, [g.mean() for g in groups], [g.std() for g in groups]])
