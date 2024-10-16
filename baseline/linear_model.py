# linear_model.py

import torch
from torch import nn

class LinearModel1(nn.Module):
    def __init__(self, C:int,  L : int, H: int, L1: int):
        """
        Takes in input of shape (B, T, C, L, H) and returns output of shape (B, 20, L, H)
        """

        self.lin1 = nn.Linear(L*H, L1) # can optionally have mlp or conv here
        self.lin2 = nn.Linear(C, 20)
        self.lin3 = nn.Linear(L1, L*H)
        self.L = L
        self.H = H
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        

        # (B, T, C, L, H) -> (B, C, L, H)
        x = x.median(1)
        # (B, C, L, H) -> (B, C, L*H)
        x = x.flatten(start_dim=-2 )

        # (B, C, L*H) -> (B, C, L1)
        x = self.lin1(x)

        x = torch.transpose(x, 1, 2)
        # (B, L1, C)  -> (B, L1, 20)
        x = self.lin2(x)

        # (B, L1, 20) -> (B, 20, L1)
        x = torch.transpose(x, 1, 2) 

        # (B, 20, L1) -> (B, 20, L*H)
        x = self.lin3(x)

        # (B, 20, L*H) -> (B, 20, L, H)
        x = x.view(x.shape[0], 20, self.L, self.H)

        return x


        