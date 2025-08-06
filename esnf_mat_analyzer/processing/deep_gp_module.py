import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeepGPModule(nn.Module):
    """
    A 1D U-Net-like model to predict geometric progression parameters from noisy tick marks.
    The model takes a 1D signal representing the locations of detected ticks and outputs
    a refined signal, from which GP parameters can be extracted.
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(DeepGPModule, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # The final layers to regress the 3 GP parameters (m0, m1, r)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # m0, m1, r
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepGPModule.

        Args:
            x: Input tensor of shape (N, C, L), where N is batch size,
               C is number of channels (1 for our signal), and L is signal length.
               The signal should be a binary vector with 1s at tick locations.

        Returns:
            A tensor of shape (N, 3) with the predicted GP parameters (m0, m1, r).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # The output of the U-Net is a refined signal.
        # We can now use a regression head on this signal to get the GP parameters.
        gp_params = self.regressor(x)
        return gp_params

    def predict_gp_params(self, noisy_marks: torch.Tensor) -> torch.Tensor:
        """
        A convenience method for prediction.

        Args:
            noisy_marks: A tensor representing the noisy tick marks.

        Returns:
            The predicted geometric progression parameters (m0, m1, r).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(noisy_marks)

if __name__ == '__main__':
    # Example of how to use the model
    # This is for demonstration and debugging purposes.

    # Create a dummy input signal (e.g., 1024 pixels wide)
    # This represents the locations of ruler ticks.
    # Batch size = 1, channels = 1, length = 1024
    dummy_input = torch.randn(1, 1, 1024)

    # Add some "ticks"
    dummy_input[0, 0, 100] = 1
    dummy_input[0, 0, 200] = 1
    dummy_input[0, 0, 300] = 1
    dummy_input[0, 0, 400] = 1

    # Instantiate the model
    model = DeepGPModule(n_channels=1, n_classes=1)

    # Get the model's prediction
    predicted_params = model.predict_gp_params(dummy_input)

    print(f"Model instantiated successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Predicted GP parameters (m0, m1, r): {predicted_params}")
    print(f"Output shape: {predicted_params.shape}")
