import torch
from .hilbert import hilbert_torch

def deconv_loss(lsparse, lphase, ltv, samples, weights):
    # create complex abs weights
    complex_abs_weights = torch.view_as_complex(weights).abs()

    # compute all losses
    sparsity_loss = lsparse * torch.mean(complex_abs_weights)
    est_wfm = weights[:, :, 1].squeeze()
    complex_wfm = hilbert_torch(est_wfm)
    est_wfm_angle = torch.angle(complex_wfm)
    phase_loss = lphase * ((torch.mean(torch.abs(torch.cos(est_wfm_angle[:, 1:]) -
                                                        torch.cos(est_wfm_angle[:, :-1])))
                + torch.mean(torch.abs(torch.sin(est_wfm_angle[:, 1:]) -
                                                    torch.sin(est_wfm_angle[:, :-1])))))

    tv_loss = ltv * torch.mean(torch.abs(est_wfm[:, 1:] - est_wfm[:, :-1]))
    loss = torch.nn.functional.mse_loss(weights,
                                        samples,
                                        reduction='mean') 
    
    total_loss = loss + phase_loss + tv_loss + sparsity_loss 

    return (
        sparsity_loss,
        phase_loss,
        tv_loss,
        loss,
        total_loss
    )

