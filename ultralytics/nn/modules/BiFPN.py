import torch.nn as nn

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN, self).__init__()
        a = in_channels[0]
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Define an upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 使用 'nearest' 上采样

    def forward(self, x):
        p3, p4, p5,*extra= x
        # Initial transformations
        p3_out = self.conv1(p3)
        p4_out = self.conv2(p4)
        p5_out = self.conv3(p5)

        # Feature fusion
        p4_out += self.upsample(p5_out)
        p3_out += self.upsample(p4_out)

        # Depthwise separable convolution
        p3_out = self.depthwise_conv(p3_out)
        p3_out = self.pointwise_conv(p3_out)
        return p3_out

