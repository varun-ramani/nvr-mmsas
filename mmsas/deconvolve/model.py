from torch import nn
import torch
import torch.nn.functional as F
from .hilbert import hilbert_torch
from .data import train_test_ds
from .functions import deconv_loss
from torch.utils.data import DataLoader

class ComplexMLP(nn.Module):
    def __init__(self):
        super(ComplexMLP, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)  # Outputs real and imaginary parts for two values

        rng = torch.Generator('cpu')
        rng.manual_seed(0)

        nn.init.xavier_uniform_(self.fc1.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc2.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc3.weight, generator=rng)
        nn.init.xavier_uniform_(self.fc4.weight, generator=rng)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)

        x = self.__normalize(x)

        return x
    
    def __normalize(self, x):
        means = x.mean(dim=1, keepdim=True)
        max_val = (x - means).abs().max(dim=1, keepdim=True).values
        x = (x - means) / max_val.clamp(min=1e-6)
        return x
    
def phase_locked_model(model, input_data):
    """
    Evaluates the model on the real waveform, computes the hilbert transform, and views back into real data.
    """
    real_data = input_data[:, :, 0].unsqueeze(2)
    model_res = model(real_data).squeeze()
    augmented_res = hilbert_torch(model_res)
    real_res = torch.view_as_real(augmented_res)
    return real_res

def train_model(data, epochs=20, batch_size=10, rng=None, lr=1e-3, lphase=1e-4, ltv=1e-4, lsparse=2e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rng is None:
        rng = torch.Generator()

    model = ComplexMLP().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Loss history for plotting
    total_loss_history = []
    sparsity_loss_history = []
    tv_loss_history = []
    mse_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        print(f"starting epoch {epoch}")

        train_ds, test_ds = train_test_ds(data, rng=rng)
        train_loader, test_loader = (
            DataLoader(train_ds, batch_size=batch_size, generator=rng),
            DataLoader(test_ds, batch_size=batch_size, shuffle=True, generator=rng)
        )

        model.train()
        for batch_num, samples_batch in enumerate(train_loader):
            optimizer.zero_grad()
            samples_batch = samples_batch.to(device)

            # load in new sample and feed into deconv model
            weights = phase_locked_model(model, samples_batch)

            # compute losses
            (sparsity_loss,
             phase_loss,
             tv_loss,
             loss,
             total_loss) = deconv_loss(
                 lsparse,
                 lphase,
                 ltv,
                 samples_batch,
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
            test_batch = next(iter(test_loader)).to(device)

            test_weights = phase_locked_model(model, test_batch)

            # compute losses
            (_,
             _,
             _,
             _,
             test_loss) = deconv_loss(
                 lsparse,
                 lphase,
                 ltv,
                 test_batch,
                 test_weights
             )
            
            test_loss_history.append(test_loss.item())

    return model, total_loss_history, sparsity_loss_history, tv_loss_history, mse_loss_history, test_loss_history, test_batch.detach().cpu(), test_weights.detach().cpu()