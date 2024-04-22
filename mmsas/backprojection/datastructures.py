import torch
from .utils import device

def get_voxel_coordinates(x_start=-50, x_end=51, y_start=-50, y_end=51, z_start=90, z_end=221, d=1):
    x = torch.arange(x_start, x_end, d) # -50 to 50 inclusive
    y = torch.arange(y_start, y_end, d) # -50 to 50 inclusive
    z = torch.arange(z_start, z_end, d) # 90 to 220 inclusive

    base_prod_res = torch.cartesian_prod(
        z, x, y
    )

    permuted_prod_res = base_prod_res[:, [1, 2, 0]]

    return permuted_prod_res

def get_angles_matrix(coords, n_rotor_step, n_actuator_step):
    coords = coords.view(n_rotor_step, n_actuator_step, 3)
    angles = torch.atan2(coords[:, :, 1], coords[:, :, 0])
    initial_angle = angles[0, :]
    relative_angles = angles - initial_angle
    relative_angles = torch.remainder(relative_angles, 2 * torch.pi)
    return relative_angles

def get_measurement_grid(marker_locs, radius, n_rotor_step, n_actuator_step):
    sensor_marker_height = marker_locs[:, 2, 3]

    # we do this in order to preserve column-major order, as per MATLAB
    sensor_marker_height = sensor_marker_height.view(-1, n_rotor_step).T

    heights = torch.mean(sensor_marker_height, dim=0)
    heights = heights - 15

    # table1 = marker_locs[:, :, 1]
    # angles1 = get_angles_matrix(table1, n_rotor_step, n_actuator_step)

    # table2 = marker_locs[:, :, 0]
    # angles2 = get_angles_matrix(table2, n_rotor_step, n_actuator_step)

    # final_angles = (angles1 + angles2) / 2

    final_angles = torch.linspace(0, 2 * torch.pi - torch.pi / 720, 1440).unsqueeze(1).repeat(1, 30)

    measurement_grid = torch.zeros(len(final_angles) * len(heights), 3)
    count = 0

    heights = heights.to(device)
    final_angles = final_angles.to(device)
    measurement_grid = measurement_grid.to(device)

    for xi in range(len(heights)):
        for theta in range(len(final_angles)):
            x = radius * torch.cos(final_angles[theta, xi])
            y = radius * torch.sin(final_angles[theta, xi])
            z = heights[xi]
            measurement_grid[count, :] = torch.tensor([x, y, z])
            count += 1

    return measurement_grid