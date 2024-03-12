#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import itertools
import json

from pathlib import Path
    
# dev = 'cpu:0'


# In[2]:

output_index = len(list(Path('output').glob('*'))) + 1
output_dir = f'./output/{output_index}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def hilbert_torch(x):
    N = x.shape[-1]

    # add extra dimension that will be removed later.
    if x.ndim == 1:
        x = x[None, :]

    # Take forward fourier transform
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros_like(x)

    if N % 2 == 0:
        h[:, 0] = h[:, N // 2] = 1
        h[:, 1:N // 2] = 2
    else:
        h[:, 0] = 1
        h[:, 1:(N + 1) // 2] = 2

    # Take inverse Fourier transform
    x_hilbert = torch.fft.ifft(Xf * h.to(Xf.device), dim=-1).squeeze()

    return x_hilbert


# In[3]:

with h5py.File('run10_halfhat_deconv.mat', 'r') as file:
    data = file['outputMatrix'][()]  # Extract the dataset

# Assuming data is a structured array with 'real' and 'imag' fields
complex_data = np.empty(data.shape, dtype=np.complex64)
complex_data.real = data['real']
complex_data.imag = data['imag']

# Convert to a PyTorch tensor of complex type
tensor_3d = torch.view_as_complex(torch.from_numpy(complex_data.view(np.float32).reshape(data.shape + (2,))))

# reshape it and split back into real
data = tensor_3d
data = data.permute(1, 2, 0)
data = data.reshape(240*15, 1200)

# normalize the real and imaginary components separately 
real_data = torch.view_as_real(data)
# mins = real_data.min(axis=1).values
# maxes = real_data.max(axis=1).values
# normalized_data = (
#     (real_data - mins[:, np.newaxis, :]) 
#     / (maxes - mins)[:, np.newaxis, :]
# )

normalized_data = real_data

from sklearn.model_selection import train_test_split

# Assuming 'samples' is your dataset
samples, test_samples = train_test_split(normalized_data, test_size=0.2, random_state=42)
#samples = samples.T
#test_samples = test_samples.T
#print(np.shape(samples))


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComplexMLP(nn.Module):
    def __init__(self):
        super(ComplexMLP, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)  # Outputs real and imaginary parts for two values

        rng = torch.Generator(device)
        rng.manual_seed(0)

        nn.init.xavier_uniform_(self.fc1.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc2.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc3.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc4.weight, generator=rng)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def compute_loss(lsparse, lphase, ltv, samples, weights):
    # create complex abs weights
    complex_abs_weights = torch.view_as_complex(weights).abs()

    # compute all losses
    sparsity_loss = lsparse * torch.mean(complex_abs_weights)
    est_wfm = np.squeeze(weights[:,:,1])
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def try_hyperparameter_combo(lr=1e-3, lphase=1e-4, ltv=1e-4, lsparse=2e-2):
    model = ComplexMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trans = list(divide_chunks(list(range(0, np.shape(samples)[0])), 10))

    # Early stopping parameters
    early_stopping_patience = 20
    min_delta = 1e-9  # Minimum change to quantify as an improvement
    best_loss = float('inf')
    patience_counter = 0

    # Loss history for plotting
    total_loss_history = []
    sparsity_loss_history = []
    tv_loss_history = []
    mse_loss_history = []
    test_loss_history = []

    for epoch in range(3):
        print(f"starting epoch {epoch}")

        model.train()
        for batch_num, trans_batch in enumerate(trans):
            optimizer.zero_grad()

            # load in new sample and feed into deconv model
            samples_batch_all = samples[trans_batch]
            weights = model(samples_batch_all)

            # compute losses
            (sparsity_loss,
             phase_loss,
             tv_loss,
             loss,
             total_loss) = compute_loss(
                 lsparse,
                 lphase,
                 ltv,
                 samples_batch_all,
                 weights
             )

            total_loss.backward()
            optimizer.step()

        total_loss_history.append(total_loss.item())
        sparsity_loss_history.append(sparsity_loss.item())
        tv_loss_history.append(tv_loss.item())
        mse_loss_history.append(loss.item())
                
        model.eval()
        with torch.no_grad():
            # feed all test samples into model
            test_weights = model(test_samples)

            # compute losses
            (_,
             _,
             _,
             _,
             test_loss) = compute_loss(
                 lsparse,
                 lphase,
                 ltv,
                 samples_batch_all,
                 weights
             )
            
            test_loss_history.append(test_loss.item())

    return total_loss_history, sparsity_loss_history, tv_loss_history, mse_loss_history, test_loss_history, test_samples.detach().cpu(), test_weights.detach().cpu()

def plot_histories(output_dir, total_loss_history, sparsity_loss_history, tv_loss_history, mse_loss_history, test_loss_history, test_samples, test_weights):
    output_path = Path(output_dir)
    output_path.mkdir()

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss')
    plt.plot(sparsity_loss_history, label='Sparsity Loss')
    plt.plot(tv_loss_history, label='TV Loss')
    plt.plot(mse_loss_history, label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Components Over Epochs')
    plt.savefig(output_path / 'loss_components_over_epochs.png')


    plt.figure(figsize=(10, 6))
    plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Test Loss Over Epochs')
    plt.savefig(output_path / 'test_loss_over_epochs.png')
    plt.close()

    complex_abs_samples = torch.view_as_complex(test_samples).abs()[0, :]
    complex_abs_weights = torch.view_as_complex(test_weights).abs()[0, :]
    samples = test_samples[0, :, 0]
    weights = test_weights[0, :, 0]

    signals = [
        ("Input Signal (Real)", samples),
        ("Processed Signal (Real)", weights),
        ("Input Signal (Complex Abs)", complex_abs_samples),
        ("Processed Signal (Complex Abs)", complex_abs_weights)
    ]

    plt.figure(figsize=(10, 10))

    for i, (title, signal) in enumerate(signals):
        plt.subplot(2, 2, i + 1)
        plt.plot(signal.numpy(), label=title)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'signal_result.png')
    plt.close()

def grid_search_hyperparameters(lrs, lsparses, lphases, ltvs):
    combos = list(itertools.product(lrs, lsparses, lphases, ltvs))
    configs = {}
    for combo, (lr, lsparse, lphase, ltv) in enumerate(combos):
        combo_name = f'combo{combo}'
        configs[combo_name] = {
            "hyperparameters": (lr, lsparse, lphase, ltv)
        }

        print(f"[{combo + 1} / {len(combos)}]\tTrying {lr}\t{lsparse}\t{lphase}\t{ltv}")

        (total_loss_history, 
        sparsity_loss_history, 
        tv_loss_history, 
        mse_loss_history, 
        test_loss_history,
        test_samples,
        weights) = try_hyperparameter_combo(lr, lphase, ltv, lsparse)

        final_loss = torch.tensor(total_loss_history[-5:]).mean().item()
        configs[combo_name]['final_loss'] = final_loss

        plot_histories(
            Path(output_dir) / combo_name, 
            total_loss_history,
            sparsity_loss_history,
            tv_loss_history,
            mse_loss_history,
            test_loss_history,
            test_samples,
            weights
        )

        print(final_loss)

    with open(Path(output_dir) / 'configs.json', 'w') as cf:
        json.dump(configs, cf, indent=4)
    
if __name__ == "__main__":
    grid_search_hyperparameters(
        # lrs=[0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1],
        lrs=[1e-2],
        # lphases=[1e-4, 1e-3, 1e-2, 1e-1],
        lphases=[1e-1, 1e3, 1e4, 1e5],
        # ltvs=[1e-4, 1e-3, 1e-2, 1e-1],
        ltvs=[1e-1, 1e3, 1e4, 1e5],
        lsparses=[1, 1.25, 1.5]
    )
        

# %%
