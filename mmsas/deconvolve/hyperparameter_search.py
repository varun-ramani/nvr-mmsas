#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch 
import os
import matplotlib.pyplot as plt
import itertools
import json
from .model import train_model
import requests

from pathlib import Path
    
output_index = len(list(Path('output').glob('*'))) + 1
output_dir = f'./output/{output_index}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# In[4]:

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

    complex_abs_samples = torch.view_as_complex(test_samples).abs()[0]

    complex_abs_weights = torch.view_as_complex(test_weights).abs()[0]

    samples_real = test_samples[0, :, 0]
    samples_imag = test_samples[0, :, 1]
    weights_real = test_weights[0, :, 0]
    weights_imag = test_weights[0, :, 1]

    # print(torch.nn.functional.mse_loss(samples_real, weights_real))
    # print(torch.nn.functional.mse_loss(samples_imag, weights_imag))

    signals = [
        ("Input Signal (Real)", samples_real),
        ("Processed Signal (Real)", weights_real),
        ("Input Signal (Imaginary)", samples_imag),
        ("Processed Signal (Imaginary)", weights_imag),
        ("Input Signal (Complex Abs)", complex_abs_samples),
        ("Processed Signal (Complex Abs)", complex_abs_weights)
    ]

    plt.figure(figsize=(10, 10))

    for i, (title, signal) in enumerate(signals):
        plt.subplot(2, 3, i + 1)
        plt.plot(signal.numpy(), label=title)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'signal_result.png')
    plt.close()

def log_to_webhook(message):
    webhook = 'https://discord.com/api/webhooks/1217693168234008606/VxbqGnx5jeo7WVO-futht66b3WNHYV25ifJNI8Slkky8ZmPvIVqzLQCg2VKCTzH0cFi0'
    requests.post(webhook, {
        'content': message
    })

def grid_search_hyperparameters(data, lrs, lsparses, lphases, ltvs, epochs=20, batch_size=10):
    combos = list(itertools.product(lrs, lsparses, lphases, ltvs))
    configs = {}
    for combo, (lr, lsparse, lphase, ltv) in enumerate(combos):
        combo_name = f'combo{combo}'
        configs[combo_name] = {
            "hyperparameters": (lr, lsparse, lphase, ltv)
        }

        msg = (f"[{combo + 1} / {len(combos)}]\tTrying {lr}\t{lsparse}\t{lphase}\t{ltv}")
        print(msg)
        log_to_webhook(msg)

        rng = torch.Generator()
        rng.manual_seed(0)

        (model,
         total_loss_history, 
         sparsity_loss_history, 
         tv_loss_history, 
         mse_loss_history, 
         test_loss_history,
         test_samples,
         weights) = train_model(data, epochs, batch_size, rng, lr, lphase, ltv, lsparse)

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

    with open(Path(output_dir) / 'configs.json', 'w') as cf:
        json.dump(configs, cf, indent=4)
    
if __name__ == "__main__":
    grid_search_hyperparameters(
        'run10_halfhat_deconv.mat',
        lrs=[0.5e-2],
        # lphases=[1, 1e2, 1e4],
        # ltvs=[1, 1e2, 1e4],
        # ltvs=[1e-2, 1e-1, 1, 1e1, 1e2],
        # lphases=[1e-2, 1e-1, 1, 1e1, 1e2],
        lsparses=[0.4],
        lphases=[0],
        ltvs=[0],
        epochs=2,
        batch_size=20
    )
        

# %%
