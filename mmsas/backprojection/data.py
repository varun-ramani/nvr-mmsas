from dataclasses import dataclass, fields
import torch
from scipy.io import loadmat

@dataclass
class ViconData:
    sar_data: torch.Tensor
    frequency: torch.Tensor
    rx_ant_pos: torch.Tensor
    tx_ant_pos: torch.Tensor
    z_target_radius: torch.Tensor
    marker_locs: torch.Tensor
    device: torch.device = torch.device('cpu')

    @property
    def n_rx(self):
        return self.sar_data.shape[0]

    @property
    def n_tx(self):
        return self.sar_data.shape[1]

    @property
    def n_actuator_step(self):
        return self.sar_data.shape[2]

    @property
    def n_rotor_step(self):
        return self.sar_data.shape[3]

    @property
    def n_sample(self):
        return self.sar_data.shape[4]
    
    def to(self, device):
        """
        Put all fields onto the provided torch device and return a new ViconData object.
        """

        if type(device) is not torch.device:
            device = torch.device(device)

        kwargs = {
            'device': device
        }

        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device)
            kwargs[field.name] = value

        return ViconData(**kwargs)
    
    @staticmethod
    def read(source_file):
        data = loadmat(source_file)

        # unpack data
        sar_data = torch.tensor(data['rawDataCal'], dtype=torch.complex128)
        frequency = torch.tensor(data['frequency'], dtype=torch.float64)
        rx_ant_pos = torch.tensor(data['rxAntPos'], dtype=torch.float64)
        tx_ant_pos = torch.tensor(data['txAntPos'], dtype=torch.float64)
        z_target_radius = torch.tensor(data['zTarget_radius'], dtype=torch.float64)
        recentered_marker_locs = torch.tensor(data['recentered_marker_locs'], dtype=torch.float64)

        vicon_data = ViconData(
            sar_data=sar_data,
            frequency=frequency,
            rx_ant_pos=rx_ant_pos,
            tx_ant_pos=tx_ant_pos,
            z_target_radius=z_target_radius,
            marker_locs=recentered_marker_locs
        )
    
        if frequency.numel() == 4 and vicon_data.n_sample > 1:
            # In PyTorch, we don't have cell arrays, so we handle this with normal tensors
            f0, K, fS, adcStart = frequency.squeeze()
            f0 = f0 + adcStart * K  # This is for ADC sampling offset
            f = f0 + torch.arange(vicon_data.n_sample) * K / fS  # wideband frequency
        elif frequency.numel() == 1 and vicon_data.n_sample == 1:
            f = frequency.item()  # single frequency
        else:
            raise ValueError('Please correct the configuration and data for 3D processing')
        
        vicon_data.frequency = f

        return vicon_data

