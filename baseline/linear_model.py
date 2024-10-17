# linear_model.py

import torch
from torch import nn

class LinearModel1(nn.Module):
    def __init__(self, C: int, W: int, H: int, L1: int, nb_classes: int):
        """
        Takes in input of shape (B, T, C, W, H) and returns output of shape (B, nb_classes, W, H)
        """
        super(LinearModel1, self).__init__()
        self.W = W
        self.H = H
        self.nb_classes = nb_classes
        
        # Flatten the spatial dimensions
        self.flatten = nn.Flatten(start_dim=-2)  # Flatten W and H into one dimension
        
        # Define linear layers
        self.lin1 = nn.Linear(W * H, L1)  # From flattened spatial dimensions to L1
        self.lin2 = nn.Linear(C, nb_classes)  # From channels to number of classes
        self.lin3 = nn.Linear(L1, W * H)  # From L1 back to flattened spatial dimensions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, W, H) -> (B, C, W, H) via median over T
        x = torch.median(x, dim=1).values  # Corrected dim to 1 for temporal dimension
        print("after median", x.shape)
        # Flatten the spatial dimensions
        x = self.flatten(x)  # (B, C, W*H)
        
        # Apply the first linear layer
        x = self.lin1(x)  # (B, C, L1)        
        
        # Transpose to (B, L1, C) to apply lin2 across channels
        x = x.transpose(1, 2)  # (B, L1, C)
        
        # Apply the second linear layer
        x = self.lin2(x)  # (B, L1, nb_classes)
        
        # Transpose to (B, nb_classes, L1)
        x = x.transpose(1, 2)  # (B, nb_classes, L1)
        
        # Apply the third linear layer
        x = self.lin3(x)  # (B, nb_classes, W*H)
        
        # Reshape back to (B, nb_classes, W, H)
        x = x.view(x.shape[0], self.nb_classes, self.W, self.H)  # (B, nb_classes, W, H)
        print(x.shape)
        return x
