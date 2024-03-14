import torch
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from .data import train_test_ds

def hilbert_torch(signals):
    # Get the shape of the input tensor
    M, N = signals.shape

    # Perform the FFT along the last axis (N)
    fft_signals = torch.fft.fft(signals, dim=-1)

    # Create a filter for the Hilbert transform in the frequency domain
    h = torch.zeros(N, dtype=torch.float)
    if N % 2 == 0:
        # For even length signals
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        # For odd length signals
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    # Apply the filter to each signal in the batch
    h = h.to(signals.device)  # Ensure the filter is on the same device as the input
    fft_filtered = fft_signals * h

    # Perform the inverse FFT
    hilbert_transform = torch.fft.ifft(fft_filtered, dim=-1, norm='backward')

    return hilbert_transform

# Example usage
# signals = torch.randn(10, 1200)  # Replace this with your tensor of shape [10, 1200]
# hilbert_transformed_signals = hilbert_torch(signals)


if __name__ == "__main__":
    train_ds, test_ds = train_test_ds('run10_halfhat_deconv.mat')
    
    train_loader = DataLoader(train_ds, pin_memory=True)
    test_loader = DataLoader(test_ds, pin_memory=True)

    test_samples = torch.vstack(list(test_loader))
    test_samples_real = test_samples[:, :, 0]
    test_samples_complex = test_samples[:, :, 1]
    complex_signals = hilbert_torch(test_samples_real)

    # is_equal = (test_samples_real[0] == complex_signals[0].real).sum()

    plt.figure()
    sns.lineplot(test_samples_complex[0])
    plt.figure()
    sns.lineplot(complex_signals[0].imag)
    plt.figure()
    sns.lineplot(complex_signals[0].abs())
    plt.show()