import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import C3k2
from .conv import Conv


class SEWeight(nn.Module):
    """
    Squeeze-and-Excitation Weight Module
    保持原样，这部分没有兼容性问题
    """

    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.pool(x))  # (B,C,1,1) in [0,1]


class LearnableHighFreqDW(nn.Module):
    """
    【改进版高频特征提取器】
    1. 使用可学习的卷积替代固定的 Haar 小波，精度更高。
    2. 使用 padding_mode='replicate' 解决边界噪声问题，且 ONNX 兼容。
    """

    def __init__(self, c1):
        super().__init__()
        # 使用 3x3 Depthwise Convolution
        # groups=c1 保证是逐通道提取特征 (类似 Haar 的逻辑)
        # padding_mode='replicate' 完美复刻了原有 replicate 填充的精度优势
        self.dw_conv = nn.Conv2d(
            c1, c1,
            kernel_size=3, stride=1, padding=1,
            groups=c1, bias=False,
            padding_mode='replicate'
        )
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输出维度: (B, c1, H, W)
        return self.act(self.bn(self.dw_conv(x)))


class DSFM(nn.Module):
    def __init__(self, c1, c2, shortcut=False, e=0.25, *args, **kwargs):
        super().__init__()
        self.main = C3k2(c1, c2, shortcut=shortcut, e=0.25)

        # 1. 替换为可学习的高频提取模块
        self.haar = LearnableHighFreqDW(c1)

        # 2. 调整通道数
        # 原版 Haar 输出是 3*c1，现在是 c1 (去除了冗余)
        # 所以这里的输入通道从 3*c1 改为 c1
        self.haar_process = Conv(c1, c1 // 2, k=1, s=1)

        self.fuse = Conv(c2 + c1 // 2, c2, k=1, s=1)

        # 高频分支 SE（通道是 c1//2）
        self.h_se = SEWeight(c1 // 2, r=8)
        self.att_gain = nn.Parameter(torch.zeros(1))  # alpha=0 起步

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 主分支
        m = self.main(x)

        # 高频分支 (可学习，自动处理边界 Padding)
        h = self.haar(x)
        h = self.haar_process(h)

        #  Residual gating on h
        w = self.h_se(h)  # (B,c1//2,1,1)

        # 这一步计算完全支持 ONNX及后续格式转换
        h = h * (1.0 + self.att_gain * w)

        # 融合
        y = torch.cat([m, h], dim=1)
        y = self.fuse(y)

        if self.add:
            return x + y
        return y
