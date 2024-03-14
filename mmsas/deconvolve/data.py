from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import numpy as np
import torch

class DeconvDataset(Dataset):
    def __init__(self, ds_file, with_norm=True, with_positive_norm=False, keep_complex=False):
        self.ds_file = ds_file
        with h5py.File(ds_file, 'r') as file:
            dataset = file['outputMatrix'][()]  # Extract the dataset

        complex_data = np.empty(dataset.shape, dtype=np.complex64)
        complex_data.real = dataset['real']
        complex_data.imag = dataset['imag']

        # Convert to a PyTorch tensor of complex type
        tensor_3d = torch.view_as_complex(torch.from_numpy(complex_data.view(np.float32).reshape(dataset.shape + (2,))))

        # reshape the data
        data = tensor_3d
        data = data.permute(1, 2, 0)
        data = data.reshape(240*15, 1200)

        real_data = torch.view_as_real(data)
        self.data = real_data
        if with_norm:
            if with_positive_norm:
                self.__positive_normalize()
            else:
                self.__zero_center_normalize()

        if keep_complex:
            self.__keep_data_complex()

    def __keep_data_complex(self):
        self.data = torch.view_as_complex(self.data)

    def __positive_normalize(self):
        """
        Normalizes data to occur within [0, 1]
        """
        mins = self.data.min(axis=1).values
        maxes = self.data.max(axis=1).values

        normalized_data = (
            (self.data - mins[:, np.newaxis, :]) 
            / (maxes - mins)[:, np.newaxis, :]
        )
        self.data = normalized_data

    def __zero_center_normalize(self):
        """
        Normalizes data to be centered at 0 and occur within [-1, 1]
        """
        means = self.data.mean(axis=1)
        amplitudes = self.data.max(axis=1).values
        normalized_data = (
            (self.data - means[:, np.newaxis, :]) / 
            amplitudes[:, np.newaxis, :]
        )
        self.data = normalized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]
    
def train_test_ds(ds_file, test_fraction=0.2, rng=None):
    if rng is None:
        rng = torch.Generator()

    base_ds = DeconvDataset(ds_file)
    train_ds, test_ds = random_split(
        dataset=base_ds, 
        lengths=[1 - test_fraction, test_fraction], 
        generator=rng
    )

    return train_ds, test_ds

if __name__ == "__main__":
    # tests
    train_ds, test_ds = train_test_ds('run10_halfhat_deconv.mat')
    
    train_loader = DataLoader(train_ds, pin_memory=True)
    test_loader = DataLoader(test_ds, pin_memory=True)