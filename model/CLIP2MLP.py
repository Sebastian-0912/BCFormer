import torch.nn as nn
# Model Definition
class CLIP2MLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CLIP2MLP, self).__init__()
        self.l1 = nn.Linear(512, hidden_dim)  # Text (512) + Image (512)
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, combined_embeddings):
        x = self.l1(combined_embeddings)
        x = self.relu(x)
        x = self.l2(x)
        return x