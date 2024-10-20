import torch
import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleSegmentationModel, self).__init__()

        # A very basic architecture: Encoder + Decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Input x shape: (B, Channels, H, W)
        x = self.encoder(x)
        x = self.decoder(x)
        # Output x shape: (B, Classes, H, W)
        return x

class SimpleSegmentationModelWrapper(nn.Module):
    """
    Wraps around the SimpleSegmentationModel
    to go from (B,T,in_channels,L,H) -> (B,T,out_channels,L,H)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_model = SimpleSegmentationModel(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        
        B, T, C, L, H = x.shape
        assert C == self.in_channels
        x = x[:,0,:,:,:]
        output = self.base_model(x) 
        return output.view(B, self.out_channels, L, H)
