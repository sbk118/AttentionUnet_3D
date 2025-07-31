import os
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resize

# 원본 데이터 폴더와 저장 폴더
input_dir = r"../data/BraTS-PEDs2024_Training"
output_dir = r"../data/downsamplings"
os.makedirs(output_dir, exist_ok=True)

# 타겟 크기 (3D 해상도)
target_shape = (128, 128, 32)

# 리사이즈 트랜스폼 정의
resize_img = Resize(spatial_size=target_shape, mode="trilinear")
resize_lbl = Resize(spatial_size=target_shape, mode="nearest")

# 모든 .nii.gz 파일 순회
for file in os.listdir(input_dir):
    if file.endswith(".nii.gz"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        print(f"🔄 처리 중: {file}")

        # nibabel 로드
        nii = nib.load(input_path)
        data = nii.get_fdata()
        affine = nii.affine

        # Torch 변환 및 차원 조정: [D, H, W] -> [1, D, H, W]
        tensor = torch.tensor(data).unsqueeze(0)

        # seg인지 판단
        is_mask = "seg" in file.lower()

        # 리사이즈 수행
        resized = resize_lbl(tensor) if is_mask else resize_img(tensor)
        resized = resized.squeeze().numpy()

        # 새 nib 파일로 저장
        resized_nii = nib.Nifti1Image(resized, affine)
        nib.save(resized_nii, output_path)
        print(f"✅ 저장 완료: {output_path}")

print("\n🎉 모든 파일 다운사이즈 및 저장 완료!")