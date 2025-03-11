# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchvision.models import ResNet34_Weights
################################################################################
#                            U-Net Building Blocks                             #
################################################################################

class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with ReLU activations.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
    Upscaling using transposed convolution followed by double conv.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, the feature map from the encoder will be concatenated,
        # so the input to the double conv is twice out_channels.
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        # x1: the decoder feature to upsample, x2: corresponding encoder feature (skip connection)
        x1 = self.up(x1)
        # Handle potential size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution to map to desired number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

################################################################################
#                           ResNet-based U-Net Model                           #
################################################################################

class ResNetUNet(nn.Module):
    """
    A U-Net architecture that uses a pretrained ResNet-34 encoder.
    This model maps a 3-channel input (RGB) to a 3-channel output (RGB).
    
    The encoder uses:
      - initial layers (conv1, bn1, relu) as x0 (features at 16x16 for a 32x32 input)
      - maxpool and subsequent ResNet layers as:
            x1: maxpool output and encoder layer1 (8x8)
            x2: encoder layer2 (4x4)
            x3: encoder layer3 (2x2)
            x4: encoder layer4 (1x1)
    The decoder upsamples these features back to the original resolution.
    """
    def __init__(self, n_channels=3, n_classes=3, pretrained=True):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load pretrained ResNet-34 and extract its layers
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.initial = nn.Sequential(
            resnet.conv1,  # 7x7 conv, stride=2 -> output: (B, 64, 16, 16) for 32x32 input
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # 3x3 max pool, stride=2 -> (B, 64, 8, 8)
        self.encoder1 = resnet.layer1  # (B, 64, 8, 8)
        self.encoder2 = resnet.layer2  # (B, 128, 4, 4)
        self.encoder3 = resnet.layer3  # (B, 256, 2, 2)
        self.encoder4 = resnet.layer4  # (B, 512, 1, 1)

        # Decoder: Upsample and combine with skip connections.
        self.up1 = Up(512, 256)  # upsample e4 and combine with e3
        self.up2 = Up(256, 128)  # combine with e2
        self.up3 = Up(128, 64)   # combine with e1
        self.up4 = Up(64, 64)    # combine with initial features from conv1 block
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)        # (B, 64, 16, 16)
        x1 = self.maxpool(x0)       # (B, 64, 8, 8)
        e1 = self.encoder1(x1)      # (B, 64, 8, 8)
        e2 = self.encoder2(e1)      # (B, 128, 4, 4)
        e3 = self.encoder3(e2)      # (B, 256, 2, 2)
        e4 = self.encoder4(e3)      # (B, 512, 1, 1)

        # Decoder with skip connections
        d1 = self.up1(e4, e3)       # (B, 256, 2, 2)
        d2 = self.up2(d1, e2)       # (B, 128, 4, 4)
        d3 = self.up3(d2, e1)       # (B, 64, 8, 8)
        d4 = self.up4(d3, x0)       # (B, 64, 16, 16)

        logits = self.outc(d4)      # (B, n_classes, 16, 16)
        # Upsample to match input size (32x32)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        return logits

# For convenience, alias ResNetUNet as UNet so existing code can remain unchanged.
UNet = ResNetUNet

if __name__ == "__main__":
    # Quick test: create a model and pass a dummy input
    model = ResNetUNet(n_channels=3, n_classes=3, pretrained=True)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # Expected: (1, 3, 32, 32)
