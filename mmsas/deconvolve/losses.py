import torch
import torch.nn.functional as F

def sparsity_loss(output):
    return torch.sum(torch.abs(output))

def tv_phase_loss(output):
    phase = torch.atan2(output[:, 3::2], output[:, 2::2])  # Imaginary, Real parts for phase calculation
    diff = phase[:, 1:] - phase[:, :-1]
    return torch.sum(torch.abs(diff))

def tv_loss(output):
#     phase = torch.atan2(output[:, 3::2], output[:, 2::2])  # Imaginary, Real parts for phase calculation
    diff = output[:, 1:] - output[:, :-1]
    return torch.sum(torch.abs(diff))

def complex_mse_loss(output, target):
    return F.mse_loss(output[:, :2], target[:, :2]) + F.mse_loss(output[:, 2:], target[:, 2:])