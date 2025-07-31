import importlib

import torch
import torch.nn as nn

from Attention_UNet_3D.segmentation.BuildingBlocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv


# 초기 채널 수와 원하는 단계 수에 따라 feature map의 크기 리스트 생성
def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class UNet3D(nn.Module):
    """
    3D U-Net 모델. 논문 "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    (https://arxiv.org/pdf/1606.06650.pdf) 을 기반으로 구현됨.

    Args:
        in_channels (int): 입력 채널 수 (예: MRI의 경우 1 또는 4)
        out_channels (int): 출력 세그멘테이션 마스크 수 (클래스 수 또는 이진 마스크 수)
        f_maps (int 또는 tuple): 인코더 각 단계에서의 feature map 수. 정수인 경우 기하급수적으로 증가함
        final_sigmoid (bool): 마지막 1x1 convolution 이후 nn.Sigmoid를 사용할지 여부.
            - 이진 분류 시 True (BCEWithLogitsLoss)
            - 다중 클래스 분류 시 False (CrossEntropyLoss)
        layer_order (str): SingleConv에서 사용할 레이어 순서 (예: 'crg'는 Conv3D -> ReLU -> GroupNorm)
        init_channel_number (int): 인코더의 첫 번째 convolution 채널 수 (기본값: 64)
        num_groups (int): GroupNorm에서 사용할 그룹 수
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        # f_maps가 정수라면, 6단계로 feature map 리스트를 생성
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=6)

        # -------------------
        # 인코더 경로 구성
        # -------------------
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                # 첫 번째 인코더는 pooling 없이 적용
                encoder = Encoder(
                    in_channels, out_feature_num,
                    apply_pooling=False,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups
                )
            else:
                # 그 외는 풀링 포함
                encoder = Encoder(
                    f_maps[i - 1], out_feature_num,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # -------------------
        # 디코더 경로 구성
        # -------------------
        decoders = []
        reversed_f_maps = list(reversed(f_maps))  # 디코더는 역순으로 구성
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]  # skip connection 포함
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(
                in_feature_num, out_feature_num,
                basic_module=DoubleConv,
                conv_layer_order=layer_order,
                num_groups=num_groups
            )
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # -------------------
        # 최종 1x1 Conv (클래스 수로 출력 채널 수 줄임)
        # -------------------
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        # 전역 평균 풀링 (압축된 feature vector 추출용)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # 최종 활성화 함수 (inference 시에만 적용)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # -------------------
        # 인코더 순방향 계산
        # -------------------
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # 디코더와 연결하기 위해 출력 저장 (skip connection)
            encoders_features.insert(0, x)  # 역순 저장

        # 최상위 인코더 출력에서 feature vector 추출
        pool_fea = self.avg_pool(encoders_features[0]).squeeze(0).squeeze(1).squeeze(1).squeeze(1)

        # 첫 번째 encoder의 출력은 디코더 입력으로 사용되지 않음 (center feature)
        encoders_features = encoders_features[1:]

        # -------------------
        # 디코더 순방향 계산
        # -------------------
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # skip connection과 디코더 입력을 결합해 업샘플링
            x = decoder(encoder_features, x)

        # 최종 conv (클래스 수로 채널 축소)
        x = self.final_conv(x)

        # 학습 시에는 로짓 그대로 출력, 추론 시에만 활성화 함수 적용
        if not self.training:
            x = self.final_activation(x)

        # 출력: segmentation 결과, 글로벌 feature vector (classification에 활용 가능)
        return x, pool_fea
