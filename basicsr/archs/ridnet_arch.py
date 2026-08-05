import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer


class MeanShift(nn.Conv2d):
    """ Data normalization with mean and std.

    Args:
        rgb_range (int): Maximum value of RGB.
        rgb_mean (list[float]): Mean for RGB channels.
        rgb_std (list[float]): Std for RGB channels.
        sign (int): For subtraction, sign is -1, for addition, sign is 1.
            Default: -1.
        requires_grad (bool): Whether to update the self.weight and self.bias.
            Default: True.
    """

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1, requires_grad=True):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = requires_grad


class EResidualBlockNoBN(nn.Module):
    """Enhanced Residual block without BN.

    There are three convolution layers in residual branch.

    It has a style of:
        ---Conv-ReLU-Conv-ReLU-Conv-+-ReLU-
         |__________________________|
    """

    def __init__(self, in_channels, out_channels):
        super(EResidualBlockNoBN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out


class MergeRun(nn.Module):
    """ Merge-and-run unit.

    This unit contains two branches with different dilated convolutions,
    followed by a convolution to process the concatenated features.

    Paper: Real Image Denoising with Feature Attention
    Ref git repo: https://github.com/saeed-anwar/RIDNet
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MergeRun, self).__init__()

        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, 2, 2), nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 3, 3), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, 4, 4), nn.ReLU(inplace=True))

        self.aggregation = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True))

    def forward(self, x):
        dilation1 = self.dilation1(x)
        dilation2 = self.dilation2(x)
        out = torch.cat([dilation1, dilation2], dim=1)
        out = self.aggregation(out)
        out = out + x
        return out


class ChannelAttention(nn.Module):
    """Channel attention.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default:
    """

    def __init__(self, mid_channels, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid_channels, mid_channels // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(mid_channels // squeeze_factor, mid_channels, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class EAM(nn.Module):
    """Enhancement attention modules (EAM) in RIDNet.

    This module contains a merge-and-run unit, a residual block,
    an enhanced residual block and a feature attention unit.

    Attributes:
        merge: The merge-and-run unit.
        block1: The residual block.
        block2: The enhanced residual block.
        ca: The feature/channel attention unit.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(EAM, self).__init__()

        self.merge = MergeRun(in_channels, mid_channels)
        self.block1 = ResidualBlockNoBN(mid_channels)
        self.block2 = EResidualBlockNoBN(mid_channels, out_channels)
        self.ca = ChannelAttention(out_channels)
        # The residual block in the paper contains a relu after addition.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.merge(x)
        out = self.relu(self.block1(out))
        out = self.block2(out)
        out = self.ca(out)
        return out

class cnnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(cnnet, self).__init__()
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
        self.conv = nn.Conv2d(in_channels=180, out_channels=1, kernel_size=1)


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
    # def __init__(self
    #              ):
    #     super(cnnet, self).__init__()
    #     # 定义一个1x1的卷积层，将180个通道压缩到1个通道
    #     self.conv = nn.Conv2d(in_channels=180, out_channels=1, kernel_size=1)
    #     # 定义一个2x2的最大池化层，将空间维度从256x256缩小到128x128
    #     # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    # def forward(self, x):
    #     # 输入x的形状为 [1, 180, 256, 256]
    #
    #
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

# @ARCH_REGISTRY.register()
# class RIDNet(nn.Module):
#     """RIDNet: Real Image Denoising with Feature Attention.
#
#     Ref git repo: https://github.com/saeed-anwar/RIDNet
#
#     Args:
#         in_channels (int): Channel number of inputs.
#         mid_channels (int): Channel number of EAM modules.
#             Default: 64.
#         out_channels (int): Channel number of outputs.
#         num_block (int): Number of EAM. Default: 4.
#         img_range (float): Image range. Default: 255.
#         rgb_mean (tuple[float]): Image mean in RGB orders.
#             Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  mid_channels,
#                  out_channels,
#                  num_block=4,
#                  up_scale=2,
#                  img_range=255.,
#                  rgb_mean=(0.4488, 0.4371, 0.4040),
#                  rgb_std=(1.0, 1.0, 1.0)):
#         super(RIDNet, self).__init__()
#
#         self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)
#         self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)
#
#         self.head = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
#         self.body = make_layer(
#             EAM, num_block, in_channels=mid_channels, mid_channels=mid_channels, out_channels=mid_channels)
#         self.tail = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # res = self.sub_mean(x)
#         res=x
#         res = self.tail(self.body(self.relu(self.head(res))))
#         res = self.add_mean(res)
#
#         out = x + res
#         return out
import torch
import torch.nn as nn
@ARCH_REGISTRY.register()
class RIDNet(nn.Module):
    """RIDNet: Real Image Denoising with Feature Attention.

    Ref git repo: https://github.com/saeed-anwar/RIDNet

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of EAM modules.
            Default: 64.
        out_channels (int): Channel number of outputs.
        num_block (int): Number of EAM. Default: 4.
        up_scale (int): Upscale factor. Default: 2.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_block=4,
                 up_scale=2,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0)):
        super(RIDNet, self).__init__()
        # self.noisepredict=cnnet()
        # self.noiseget=CNENET()
        self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)

        self.head = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            EAM, num_block, in_channels=mid_channels, mid_channels=mid_channels, out_channels=mid_channels)
        self.tail = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

        # Add an upsampling layer
        self.upscale = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)

    def forward(self, x):
        # res = self.sub_mean(x)
        res = x
        # noise=self.noisepredict(x)
        # target = torch.cat((x, noise), dim=1)
        # res=self.noiseget(target)
        res = self.tail(self.body(self.relu(self.head(res))))
        # res = self.add_mean(res)

        out = x + res

        # Apply upscaling
        out = self.upscale(out)

        return out