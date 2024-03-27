import torch
from scipy.constants import speed_of_light

def backprojection_process(vicon_data, measurement_grid, voxel_coordinates):
    l = speed_of_light / vicon_data.frequency
    k = 2 * torch.pi / l
    startval = 1
    device = vicon_data.device

    lin = torch.floor(
        torch.arange(0, measurement_grid.shape[0]) / vicon_data.n_rotor_step
    ) + startval

    rot = torch.remainder(
        torch.arange(0, measurement_grid.shape[0]), 
        vicon_data.n_rotor_step
    ) + 1

    lin = lin.to(device)
    rot = rot.to(device)

    for mg in range(measurement_grid.shape[0]):
        matched_filter = build_matched_filter(mg, voxel_coordinates, measurement_grid, vicon_data, k)


def build_matched_filter(mg, voxel_coordinates, measurement_grid, vicon_data, k):
    dist_tx = (
        1e-3 * voxel_coordinates + 
        1e-3 * measurement_grid[mg, :] +
        vicon_data.tx_ant_pos.unsqueeze(1)
    )

    dist_rx = (
        1e-3 * voxel_coordinates + 
        1e-3 * measurement_grid[mg, :] +
        vicon_data.rx_ant_pos.unsqueeze(1)
    )

    dist_tx = dist_tx.unsqueeze(1)  # Shape becomes [4, 1, 1336331, 3]
    dist_rx = dist_rx.unsqueeze(0)  # Shape becomes [1, 2, 1336331, 3]

    disttot = (
        torch.sqrt(torch.sum(dist_tx**2, dim=-1)) +
        torch.sqrt(torch.sum(dist_rx**2, dim=-1))
    ).transpose(0, 1)

    # Perform element-wise operation and reshape
    k_reshaped = k.unsqueeze(0).unsqueeze(0)
    disttot_reshaped = disttot.unsqueeze(-1)
    return torch.exp(torch.matmul(disttot_reshaped, k_reshaped) * -1j)