import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
    Upscaling + DoubleConv. If skip_in is None, we just do a plain upsample.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_in=None):
        x = self.up(x)
        if skip_in is not None:
            # Pad if there's a small mismatch
            diffY = skip_in.size()[2] - x.size()[2]
            diffX = skip_in.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip_in, x], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # --- Encoder ---
        self.initial = nn.Sequential(
            resnet.conv1,  # 32×32 -> 16×16
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # 16×16 -> 8×8
        self.encoder1 = resnet.layer1  # 8×8  -> remains 8×8
        self.encoder2 = resnet.layer2  # 4×4
        self.encoder3 = resnet.layer3  # 2×2
        self.encoder4 = resnet.layer4  # 1×1

        # --- Decoder ---
        self.up1 = Up(512, 256)  # 1×1 -> 2×2
        self.up2 = Up(256, 128)  # 2×2 -> 4×4
        self.up3 = Up(128, 64)   # 4×4 -> 8×8
        self.up4 = Up(64, 64)    # 8×8 -> 16×16
        # New final up block: 16×16 -> 32×32 (no skip, or skip from self.initial if you want)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64)
        )

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.initial(x)      # (B, 64, 16,16)
        x1 = self.maxpool(x0)     # (B, 64, 8,8)
        e1 = self.encoder1(x1)    # (B, 64, 8,8)
        e2 = self.encoder2(e1)    # (B,128, 4,4)
        e3 = self.encoder3(e2)    # (B,256, 2,2)
        e4 = self.encoder4(e3)    # (B,512, 1,1)

        # --- Decoder ---
        d1 = self.up1(e4, e3)     # 2×2
        d2 = self.up2(d1, e2)     # 4×4
        d3 = self.up3(d2, e1)     # 8×8
        d4 = self.up4(d3, x0)     # 16×16
        # Final up to 32×32 (no skip, or skip from original x if you prefer)
        d5 = self.up5(d4)         # 32×32

        logits = self.outc(d5)    # (B, n_classes, 32,32)
        return logits

# Alias
UNet = ResNetUNet

if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=3)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # (1, 3, 32, 32)
