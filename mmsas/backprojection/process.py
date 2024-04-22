import torch
from rich.progress import track
from scipy.constants import speed_of_light

def build_matched_filter(mg, voxel_coordinates, measurement_grid, vicon_data, k):
    dist_tx = (
        1e-3 * voxel_coordinates - 
        (1e-3 * measurement_grid[mg, :] +
         vicon_data.tx_ant_pos.unsqueeze(1))
    )

    dist_rx = (
        1e-3 * voxel_coordinates -
        (1e-3 * measurement_grid[mg, :] +
         vicon_data.rx_ant_pos.unsqueeze(1))
    )

    dist_tx = dist_tx.unsqueeze(1)  
    dist_rx = dist_rx.unsqueeze(0)  

    disttot = (
        torch.sqrt(torch.sum(dist_tx**2, dim=-1)) +
        torch.sqrt(torch.sum(dist_rx**2, dim=-1))
    ).transpose(0, 1)

    # Perform element-wise operation and reshape
    k_reshaped = k.unsqueeze(0).unsqueeze(0)
    disttot_reshaped = disttot.unsqueeze(-1)
    
    matched_filter = torch.exp(torch.matmul(disttot_reshaped, k_reshaped) * -1j)

    return matched_filter

def backprojection_loop(vicon_data, measurement_grid, voxel_coordinates):
    l = speed_of_light / vicon_data.frequency
    k = 2 * torch.pi / l

    lin = torch.floor(
        torch.arange(0, measurement_grid.shape[0]) / vicon_data.n_rotor_step
    ).type(torch.int64)

    rot = torch.remainder(
        torch.arange(0, measurement_grid.shape[0]), 
        vicon_data.n_rotor_step
    ).type(torch.int64)

    sar_image = torch.zeros(voxel_coordinates.shape[0], 1, dtype=torch.complex64).to(vicon_data.device)

    for mg in track(list(range(measurement_grid.shape[0]))):
        matched_filter = build_matched_filter(mg, voxel_coordinates, measurement_grid, vicon_data, k)
        sar_data_indexed = vicon_data.sar_data[:, :, lin[mg], rot[mg], :].unsqueeze(3)
        val = torch.matmul(matched_filter, sar_data_indexed).sum(0).sum(0)
        sar_image += val

    return sar_image

