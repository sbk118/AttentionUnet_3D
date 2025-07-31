import os
import numpy as np
import nibabel as nib
import csv

# ----------- 설정 -----------
prediction_dir = "../predictions"
output_csv = "../predictions/volume_comparison.csv"

volume_data = []

# ----------- prediction 파일 순회 -----------
for fname in sorted(os.listdir(prediction_dir)):
    if not fname.endswith(".nii.gz"):
        continue

    case_id = fname.replace(".nii.gz", "")
    file_path = os.path.join(prediction_dir, fname)

    try:
        pred_nii = nib.load(file_path)
        pred_data = pred_nii.get_fdata().astype(np.uint8)
        spacing = pred_nii.header.get_zooms()  # (x, y, z)
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    except Exception as e:
        print(f"❌ {case_id} 로딩 실패: {e}")
        continue

    unique_labels = np.unique(pred_data)
    for label in unique_labels:
        if label == 0:
            continue  # 배경 생략

        voxel_count = np.sum(pred_data == label)
        volume_voxel_based = voxel_count * 1.0  # 단순 voxel 수
        volume_phys_based = voxel_count * voxel_volume_mm3  # 실제 물리적 부피

        volume_data.append([
            case_id,
            label,
            voxel_count,
            round(volume_voxel_based, 2),
            round(volume_phys_based, 2),
            f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f}"
        ])

# ----------- CSV 저장 -----------
with open(output_csv, mode="w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "case_id", "label", "voxel_count", "volume_voxel_based_mm3",
        "volume_phys_based_mm3", "voxel_spacing"
    ])
    writer.writerows(volume_data)

print(f"✅ 결과 저장 완료: {output_csv}")
