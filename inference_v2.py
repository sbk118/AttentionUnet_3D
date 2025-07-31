import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Resize
from Attention_UNet_3D.segmentation.attention_unet_3d import UNet3D
from utils.monai_downsampling_reshape import monai_resize

# --------- 설정 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/best_attention_unet3d_add_aug_v2.pth"
testing_path = "./data/BraTS-PEDs2024_Training"
save_root = "./predictions/aug_v2"
modalities = ["t1c", "t1n", "t2f", "t2w"]

# --------- 모델 로드 ---------
model = UNet3D(in_channels=4, out_channels=5, final_sigmoid=False, f_maps=32)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# --------- 테스트 케이스 설정 ---------
case_list = sorted([
    name for name in os.listdir(testing_path)
    if os.path.isdir(os.path.join(testing_path, name)) and not name.startswith(".")
])[175:]  # 테스트셋 일부

os.makedirs(save_root, exist_ok=True)

for case_id in case_list:
    print(f"\n🔄 예측 중: {case_id}")
    case_path = os.path.join(testing_path, case_id)
    save_path = os.path.join(save_root, f"{case_id}-pred_upscaled.nii.gz")

    # 1. 이미지 불러오기
    images = []
    for mod in modalities:
        file_path = os.path.join(case_path, f"{case_id}-{mod}.nii.gz")
        if not os.path.exists(file_path):
            print(f"❌ 누락된 modality: {file_path}")
            break
        img = nib.load(file_path).get_fdata().astype(np.float32)
        images.append(img)
    else:
        images = np.stack(images, axis=0)
        ref_nii = nib.load(os.path.join(case_path, f"{case_id}-t2f.nii.gz"))
        target_shape = ref_nii.shape  # (240, 240, 155)

        # 2. 전처리
        images = monai_resize(images, is_mask=False)
        images = (images - images.mean()) / (images.std() + 1e-8)
        images_tensor = torch.from_numpy(images).unsqueeze(0).to(device)  # [1, 4, 128, 128, 32]

        # 3. 예측
        with torch.no_grad():
            logits, _ = model(images_tensor)  # [1, 5, 128, 128, 32]
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().float()  # [128, 128, 32]

        # 4. 업샘플링
        upsampler = Resize(spatial_size=target_shape, mode="trilinear")
        pred_upscaled = upsampler(pred.unsqueeze(0)).squeeze(0).byte().numpy()

        # 5. 저장
        pred_nifti = nib.Nifti1Image(pred_upscaled, affine=ref_nii.affine)
        nib.save(pred_nifti, save_path)
        print(f"✅ 저장 완료: {save_path}")

        # 6. 🔍 예측 라벨 분포 확인
        unique_classes = np.unique(pred_upscaled)
        print("📊 예측된 클래스 (np.unique):", unique_classes)
