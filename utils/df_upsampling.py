import os
import glob
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resize
from tqdm import tqdm

# 원본 데이터 폴더
input_dir = r""
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

# 타겟 해상도
target_shape = (240, 240, 155)

# Resize transform 정의 (채널 포함)
resize_img = Resize(spatial_size=target_shape, mode="trilinear")
resize_lbl = Resize(spatial_size=target_shape, mode="nearest")

# nii.gz 파일 목록
nii_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

for file_path in tqdm(nii_files, desc="복원 중..."):
    filename = os.path.basename(file_path)
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)

    # 모양 확인: [D, H, W] -> [1, D, H, W]
    if data.ndim != 3:
        print(f":exclamation: 스킵됨 (3D 아님): {filename}")
        continue
    tensor = torch.tensor(data).unsqueeze(0)  # [1, D, H, W]

    # 마스크 여부에 따라 다른 인터폴레이션
    is_mask = "_seg" in filename
    resized = resize_lbl(tensor) if is_mask else resize_img(tensor)

    # 저장
    restored = resized.squeeze(0).numpy()  # [D, H, W]
    new_img = nib.Nifti1Image(restored, affine=img.affine)
    save_path = os.path.join(output_dir, filename.replace(".nii.gz", "_resized.nii.gz"))
    nib.save(new_img, save_path)

print("\n:tada: 복원 완료!")