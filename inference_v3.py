import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import Resize
from Attention_UNet_3D.segmentation.attention_unet_3d import UNet3D
from utils.monai_downsampling_reshape import monai_resize
import torch.nn.functional as F
from eval_v2 import dice_score_v2, dice_score, iou_score

# --------- 설정 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/best_attention_unet3d_add_aug_v1.pth"
testing_path = "./data/BraTS-PEDs2024_Training"
save_root = "./predictions/augmentation"
modalities = ["t1c", "t1n", "t2f", "t2w"]
num_classes = 5

# --------- 모델 로드 ---------
model = UNet3D(in_channels=4, out_channels=num_classes, final_sigmoid=False, f_maps=32)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# --------- 테스트 케이스 설정 ---------
case_list = sorted([
    name for name in os.listdir(testing_path)
    if os.path.isdir(os.path.join(testing_path, name)) and not name.startswith(".")
])[175:]  # 테스트셋 일부

os.makedirs(save_root, exist_ok=True)

for case_id in tqdm(case_list, desc="🔍 Inference"):
    print(f"\n🔄 예측 중: {case_id}")
    case_path = os.path.join(testing_path, case_id)
    save_path = os.path.join(save_root, f"{case_id}-pred_upscaled.nii.gz")
    gt_path = os.path.join(case_path, f"{case_id}-seg.nii.gz")

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
            probs = F.softmax(logits, dim=1)  # 확률값

        # 4. 업샘플링 (소프트맥스 확률값 기준)
        upsampler = Resize(spatial_size=target_shape, mode="nearest")
        probs_up = torch.stack([upsampler(probs[0, i]) for i in range(num_classes)])  # [5, 240, 240, 155]
        pred_labels = torch.argmax(probs_up, dim=0).byte().cpu().numpy()

        # # 5. 저장
        # pred_nifti = nib.Nifti1Image(pred_labels, affine=ref_nii.affine)
        # nib.save(pred_nifti, save_path)
        # print(f"✅ 저장 완료: {save_path}")

        # 6. 평가 (GT 존재 시)
        if os.path.exists(gt_path):
            gt = nib.load(gt_path).get_fdata().astype(np.int64)
            gt_tensor = torch.tensor(gt).unsqueeze(0)  # [1, D, H, W]
            pred_soft = probs_up.unsqueeze(0)  # [1, C, D, H, W]
            pred_label = torch.tensor(pred_labels).unsqueeze(0)

            # a. Softmax 기반 Dice
            dice_v2 = dice_score_v2(pred_soft, gt_tensor, exclude_class_0=True)

            # b. Hard Dice / IoU
            hard_dice = [dice_score(pred_label, gt_tensor, cls) for cls in range(num_classes)]
            hard_iou = [iou_score(pred_label, gt_tensor, cls) for cls in range(num_classes)]

            print(f"📏 Softmax Dice (v2): {dice_v2:.4f}")
            for cls in range(num_classes):
                print(f"  🔹 Class {cls}: Dice={hard_dice[cls]:.4f}, IoU={hard_iou[cls]:.4f}")
