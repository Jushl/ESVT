import numpy as np


def VoxelGrid(events, num_bins, height, width):
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    last_stamp = events[-1][3]
    first_stamp = events[0][3]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 3] = (num_bins - 1) * (events[:, 3] - first_stamp) / deltaT
    ts = events[:, 3]
    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    pols = events[:, 2]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    voxel_grid = voxel_grid.transpose(1, 2, 0)

    return voxel_grid



