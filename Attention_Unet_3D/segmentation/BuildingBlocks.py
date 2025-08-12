import torch
from torch import nn
from torch.nn import functional as F
from Attention_UNet_3D.segmentation.sca_3d import SCA3D

# 3D Convolution 생성 함수
def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

# 'crg'와 같은 문자열에 따라 Conv, ReLU, GroupNorm 등을 조합하는 함수
def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding)))
        elif char == 'g':
            assert i > order.index('c'), 'GroupNorm MUST go after Conv3d'
            num_groups = min(num_groups, out_channels)
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            bn_channels = in_channels if is_before_conv else out_channels
            modules.append(('batchnorm', nn.BatchNorm3d(bn_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'")
    return modules

# Conv3D + 정규화 + 활성화 단일 조합 블록
class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
            self.add_module(name, module)

# Conv 블록 2개 연속 적용 (인코더/디코더 차별 처리)
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_out = max(in_channels, out_channels // 2)
            conv1_in, conv2_in, conv2_out = in_channels, conv1_out, out_channels
        else:
            conv1_in, conv1_out, conv2_in, conv2_out = in_channels, out_channels, out_channels, out_channels

        self.add_module('SingleConv1', SingleConv(conv1_in, conv1_out, kernel_size, order, num_groups))
        self.add_module('SingleConv2', SingleConv(conv2_in, conv2_out, kernel_size, order, num_groups))

# ResNet 스타일 잔차 블록
class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8):
        super(ExtResNetBlock, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size, order, num_groups)
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size, order, num_groups)
        n_order = order
        for c in 'rel': n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size, n_order, num_groups)

        if 'l' in order: self.non_linearity = nn.LeakyReLU(0.1, inplace=True)
        elif 'e' in order: self.non_linearity = nn.ELU(inplace=True)
        else: self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        return self.non_linearity(out + residual)

# 인코더: 선택적 Pooling + DoubleConv 또는 ResNet 블록
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv,
                 conv_layer_order='crg', num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        self.pooling = nn.MaxPool3d(pool_kernel_size) if apply_pooling and pool_type == 'max' else (
            nn.AvgPool3d(pool_kernel_size) if apply_pooling else None)
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)

# 디코더: 업샘플링 + SCA3D + DoubleConv
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='crg', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            in_channels = out_channels

        self.scse = SCA3D(in_channels)
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            x = F.interpolate(x, size=encoder_features.size()[2:], mode='nearest')
            x = torch.cat((encoder_features, x), dim=1)
        else:
            x = self.upsample(x)
            x += encoder_features
        x = self.scse(x)
        return self.basic_module(x)

# 마지막 Conv 블록: Conv3D + 정규화 + 활성화 + 1x1 conv로 출력 채널 축소
class FinalConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(FinalConv, self).__init__()
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))
        self.add_module('final_conv', nn.Conv3d(in_channels, out_channels, 1))
