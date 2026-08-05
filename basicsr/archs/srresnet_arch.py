from torch import nn as nn
from torch.nn import functional as F
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer

class noisepredict(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(noisepredict, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)



    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        # 中间层
        m = self.middle(p2)

        # 解码器
        u1 = self.up1(m)
        d1 = self.decoder1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.decoder2(torch.cat([u2, e1], dim=1))

        # 输出
        out = self.final(d2)
        # out=self.conv(out)
        return out

    # def __init__(self):
    #     super(noisepredict, self).__init__()
    #     # 定义一个1x1的卷积层，将180个通道压缩到1个通道
    #     self.conv = nn.Conv2d(in_channels=180, out_channels=1, kernel_size=1)
    #     # 定义一个2x2的最大池化层，将空间维度从256x256缩小到128x128
    #     # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    # def forward(self, x):
    #     # 输入x的形状为 [1, 180, 256, 256]
    #     x = self.conv(x)  # 形状变为 [1, 1, 256, 256]
    #     # x = self.pool(x)  # 形状变为 [1, 1, 128, 128]
    #     # 输出x的形状为 [1, 1, 128, 128]
    #     return x
class CNENET(nn.Module):
    def __init__(self,
                 ):
        super(CNENET, self).__init__()
        # 定义一个1x1的卷积层，将1个通道扩展到180个通道
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x):
        # 输入x的形状为 [1, 1, 256, 256]
        x = self.conv(x)
        # 输出x的形状为 [1, 180, 256, 256]
        return x


@ARCH_REGISTRY.register()
class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
