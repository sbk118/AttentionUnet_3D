import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA3D(nn.Module):
    """
    SCA3D (Spatial and Channel Attention 3D) 모듈
    - 3D 특징 맵에 대해 채널 주의력(Channel Attention)과 공간 주의력(Spatial Attention)을 모두 적용하여
      중요한 위치 및 채널 정보를 강조함.
    - 두 attention을 모두 적용한 후 입력과 더하여 residual-like 효과 제공.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()

        # 채널 attention을 위한 전역 평균 풀링 (각 채널별 평균값만 추출, shape: [B, C, 1, 1, 1])
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # 채널 attention: (B, C) → (B, C // reduction) → (B, C)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel)
        )

        # 공간 attention: 채널 차원을 1로 줄여 공간 위치별 중요도 계산 (B, C, D, H, W) → (B, 1, D, H, W)
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()  # B, C, D, H, W

        # ----------------------
        # 채널 attention
        # ----------------------
        # 1. 평균 풀링 후 (B, C, 1, 1, 1) → (B, C)
        chn_se = self.avg_pool(x).view(bahs, chs)

        # 2. FC → ReLU → FC → sigmoid (B, C)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.sigmoid(chn_se).view(bahs, chs, 1, 1, 1)

        # 3. 원본 특징 맵에 채널별 중요도를 곱함 (broadcast)
        chn_se = torch.mul(x, chn_se)

        # ----------------------
        # 공간 attention
        # ----------------------
        # 1. 채널 → 1채널로 줄여 위치별 중요도 학습
        spa_se = self.spatial_se(x)
        spa_se = torch.sigmoid(spa_se)

        # 2. 원본 특징 맵에 위치별 중요도 곱함
        spa_se = torch.mul(x, spa_se)

        # ----------------------
        # Attention 결과 통합
        # ----------------------
        net_out = spa_se + x + chn_se  # residual + 두 attention

        return net_out
