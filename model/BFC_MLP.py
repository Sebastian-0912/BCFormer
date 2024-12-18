import torch
import torch.nn as nn
import torch.fft

# Model Definition
class BFC_MLP(nn.Module):
    def __init__(self, crop_size, hidden_dim, num_classes):
        super(BFC_MLP, self).__init__()
        self.crop_size = crop_size
        self.phase_dim = crop_size * crop_size
        self.l1 = nn.Linear(512 * 2 + self.phase_dim, hidden_dim)  # Text (512) + Image (512) + Phase
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def extract_low_frequency(self, phase):
        """
        Extract low-frequency components from the phase information.
        :param phase: Input phase tensor, shape [batch, height, width].
        :param crop_size: Size of the low-frequency region to crop (center square).
        :return: Flattened low-frequency components, shape [batch, crop_size * crop_size].
        """
        batch, height, width = phase.shape
        crop_size = self.crop_size
        # Calculate cropping indices
        center_h, center_w = height // 2, width // 2
        start_h, end_h = center_h - crop_size // 2, center_h + crop_size // 2
        start_w, end_w = center_w - crop_size // 2, center_w + crop_size // 2

        # Crop the low-frequency region
        low_freq_phase = phase[:, start_h:end_h, start_w:end_w]  # Shape: [batch, crop_size, crop_size]
        # print('low freq: ',low_freq_phase.shape)
        # Flatten the low-frequency region
        low_freq_phase = low_freq_phase.reshape(batch, -1)  # Shape: [batch, crop_size * crop_size]
        # print('low freq flatten: ',low_freq_phase.shape)
        return low_freq_phase

    def forward(self, phase, combined_embeddings):
        # Extract low-frequency components from the phase
        low_freq_phase = self.extract_low_frequency(phase)  # Adjust crop_size as needed

        # Concatenate embeddings and low-frequency phase features
        combined_features = torch.cat((combined_embeddings, low_freq_phase), dim=1)

        # MLP forward pass
        x = self.l1(combined_features)
        x = self.relu(x)
        x = self.l2(x)
        return x
