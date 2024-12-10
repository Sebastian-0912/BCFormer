import torch.nn as nn

# Model Definition
class BLIP2CLIP_TextEncoder_MLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(BLIP2CLIP_TextEncoder_MLP, self).__init__()
        self.l1 = nn.Linear(512, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text_embedding):
        x = self.l1(text_embedding)
        x = self.relu(x)
        x = self.l2(x)
        return x