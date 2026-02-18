import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    Implementation of EEGNet Convolutional Neural Network, derived from Lawhern et. al (2018) "EEGNet: a compact convolutional neural network for EEG-based
    brain–computer interfaces."
    """
    def __init__(self,
                 num_channels=96,
                 num_samples=500,
                 num_classes=2,
                 F1=6,
                 D=2,
                 kernel_length=64,
                 dropout_rate=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_samples = num_samples
        F2 = F1 * D 

        # Block 1: Temporal + Depthwise Spatial Conv
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding=(0, 16 // 2), bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Final pooling + classifier
        self.avgpool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) # (B, C, T) -> (B, 1, C, T)

        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.avgpool_final(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out

    def forward_to_depthwise(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        return x