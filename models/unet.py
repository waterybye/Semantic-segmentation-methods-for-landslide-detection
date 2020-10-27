import torch
from torch.nn import functional as F
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(256, 512)
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )
        # Decode
        self.conv_decode4 = self.expansive_block(1024, 512, 256)
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def forward(self, x):
        # Encode
        # (..., width, height) => (..., width - 4, height - 4)
        encode_block1 = self.conv_encode1(x)
        # (..., width - 4, height - 4) => (..., (width - 4)/2, (height - 4)/2)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        # (..., (width - 4) / 2, (height - 4) / 2) => (..., (width - 4)/2 - 4, (height - 4)/2 - 4)
        encode_block2 = self.conv_encode2(encode_pool1)
        # (..., (width - 4)/2 - 4, (height - 4)/2 - 4) => (..., ((width - 4)/2 - 4)/2, ((height - 4)/2 - 4)/2)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        # (..., ((width - 4)/2 - 4)/2, ((height - 4)/2 - 4)/2)
        # => (..., ((width - 4)/2 - 4)/2 - 4, ((height - 4)/2 - 4)/2 - 4)
        encode_block3 = self.conv_encode3(encode_pool2)
        # (..., ((width - 4)/2 - 4)/2 - 4, ((height - 4)/2 - 4)/2 - 4)
        # => (..., (((width - 4)/2 - 4)/2 - 4) / 2, (((height - 4)/2 - 4)/2 - 4)/2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        # (..., ((width - 4)/2 - 4)/2 - 4, ((height - 4)/2 - 4)/2 - 4)
        # => (..., (((width - 4)/2 - 4)/2 - 4) / 2, (((height - 4)/2 - 4)/2 - 4)/2)
        encode_pool4 = self.conv_maxpool4(encode_block4)

        # Bottleneck
        # (..., (((width - 4)/2 - 4)/2 - 4) / 2, (((height - 4)/2 - 4)/2 - 4)/2)
        # => (..., (((width - 4)/2 - 4)/2 - 4) / 2, (((height - 4)/2 - 4)/2 - 4)/2)
        bottleneck = self.bottleneck(encode_pool4)
        # Decode
        decode_block4 = self.crop_and_concat(bottleneck, encode_block4, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        # (batch_size, in_channels, width, height) => (batch_size, out_channels, width - 4, height - 4)
        return nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        # (batch_size, in_channels, width, height) => (batch_size, out_channels, (width - 4) * 2, (height - 4) * 2)
        return nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        # (batch_size, in_channels, width, height) => (batch_size, out_channels, width - 4, height - 4)
        return nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    @staticmethod
    def crop_and_concat(upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, [-c, -c, -c, -c])
        return torch.cat((upsampled, bypass), 1)
