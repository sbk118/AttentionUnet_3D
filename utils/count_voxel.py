import os
import numpy as np
import nibabel as nib
import csv

# ----------- 설정 -----------
prediction_dir = "../predictions/weight"
output_csv = "../predictions/weight/volume_summary.csv"

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
    except Exception as e:
        print(f"❌ {case_id} 로딩 실패: {e}")
        continue

    foreground_voxel_count = np.sum(pred_data != 0)  # 배경 제외한 모든 voxel

    volume_data.append([case_id, foreground_voxel_count])

# ----------- CSV 저장 -----------
with open(output_csv, mode="w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "foreground_voxel_count"])
    writer.writerows(volume_data)

print(f"✅ 결과 저장 완료: {output_csv}")