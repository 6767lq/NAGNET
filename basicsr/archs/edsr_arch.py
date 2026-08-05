import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
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

@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # self.upconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        # self.noisepredict=cnnet()
        # self.noiseget=CNENET()

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        # noise = self.noisepredict(x)
        # target = torch.cat((x, noise), dim=1)
        # x = self.noiseget(target)



        # x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x


        x = self.conv_last(self.upsample(res))
        # x = x / self.img_range + self.mean
        # noise2=self.upconv(noise)
        # target = torch.cat((x, noise2), dim=1)
        # x = self.noiseget(target)

        return x
