import os
import torch
import numpy as np
import nibabel as nib
from Attention_UNet_3D.segmentation.attention_unet_3d import UNet3D
from utils.resize import resize_xy_and_crop_z

# --------- 설정 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/attention_unet3d.pth"
testing_path = "./data/testing/"
save_root = "./predictions/"
modalities = ["t1c", "t1n", "t2f", "t2w"]

# --------- 모델 로드 ---------
model = UNet3D(in_channels=4, out_channels=5, final_sigmoid=False, f_maps=32, layer_order='crg', num_groups=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# --------- 케이스 순회 ---------
case_list = sorted([name for name in os.listdir(testing_path) if os.path.isdir(os.path.join(testing_path, name))])
print(f"🔍 총 케이스 수: {len(case_list)}")

for case_id in case_list:
    print(f"🔄 예측 중: {case_id}")
    case_path = os.path.join(testing_path, case_id)
    save_path = os.path.join(save_root, f"{case_id}-pred.nii.gz")

    # 1. 이미지 로딩
    images = []
    for mod in modalities:
        file_path = os.path.join(case_path, f"{case_id}-{mod}.nii.gz")
        if not os.path.exists(file_path):
            print(f"❌ 누락된 modality: {file_path}")
            break
        img = nib.load(file_path).get_fdata().astype(np.float32)
        images.append(img)
    else:  # 모든 modality가 정상적으로 로딩된 경우에만 진행
        images = np.stack(images, axis=0)  # shape: [4, H, W, D]

        ref_nii = nib.load(os.path.join(case_path, f"{case_id}-t2f.nii.gz"))

        # 2. 전처리
        images = resize_xy_and_crop_z(images, target_xy=128, target_z=32, order=3)
        images = (images - images.mean()) / (images.std() + 1e-8)
        images_tensor = torch.from_numpy(images).unsqueeze(0).to(device)

        # 3. 예측
        with torch.no_grad():
            logits, _ = model(images_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 4. 저장
        os.makedirs(save_root, exist_ok=True)
        pred_nifti = nib.Nifti1Image(preds, affine=ref_nii.affine)
        nib.save(pred_nifti, save_path)

        print(f"✅ 저장 완료: {save_path}")
