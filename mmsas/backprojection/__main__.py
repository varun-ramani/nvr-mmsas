from scipy.io import loadmat
from scipy.constants import speed_of_light
import torch
from .data import ViconData
from .datastructures import get_measurement_grid, get_voxel_coordinates
from .process import backprojection_loop
import sys

vicon_data = ViconData.read(sys.argv[1])

measurement_grid = get_measurement_grid(
    vicon_data.marker_locs,
    vicon_data.z_target_radius,
    vicon_data.n_rotor_step,
    vicon_data.n_actuator_step
)

voxel_coordinates = get_voxel_coordinates(d=int(sys.argv[2]))

result = backprojection_loop(vicon_data, measurement_grid, voxel_coordinates)

torch.save(result, sys.argv[3])