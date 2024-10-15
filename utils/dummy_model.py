import torch
import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    def __init__(self, input_channels: int, nb_classes: int):
        super(SimpleSegmentationModel, self).__init__()

        # A very basic architecture: Encoder + Decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, nb_classes, kernel_size=3, padding=1),
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
    to go from (B,T,input_channels,L,H) -> (B,T,nb_classes,L,H)
    """
    def __init__(self, input_channels: int, nb_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.nb_classes = nb_classes
        self.base_model = SimpleSegmentationModel(input_channels, nb_classes)

    def forward(self, x: torch.Tensor):
        B, T, C, L, H = x.shape
        assert C == self.input_channels
        x = x[:,0,:,:,:]
        output = self.base_model(x) 
        return output.view(B, self.nb_classes, L, H)
