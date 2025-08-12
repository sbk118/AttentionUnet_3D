import torch.nn.functional as F

# Softmax 기반 Dice
# def dice_score(pred, target, epsilon=1e-6, exclude_class_0=True):
#     """
#     배경을 제외하고 dice score 산출
#     Args:
#         pred: [B, C, D, H, W] - model logits or probabilities (after softmax)
#         target: [B, D, H, W] - ground truth labels (0~C-1)
#     """
#     pred_soft = F.softmax(pred, dim=1)  # 확률화
#     num_classes = pred.shape[1]
#
#     total_dice = 0
#     classes = range(1, num_classes) if exclude_class_0 else range(num_classes)
#
#     for cls in classes:
#         pred_cls = pred_soft[:, cls]
#         target_cls = (target == cls).float()
#
#         intersect = torch.sum(pred_cls * target_cls)
#         union = torch.sum(pred_cls) + torch.sum(target_cls)
#         dice = (2. * intersect + epsilon) / (union + epsilon)
#         total_dice += dice
#
#     return total_dice / len(classes)

def dice_score(pred, target, class_idx):
    pred_bin = (pred == class_idx)
    target_bin = (target == class_idx)
    intersection = (pred_bin & target_bin).sum().item()
    union = pred_bin.sum().item() + target_bin.sum().item()
    if union == 0:
        return 1.0  # 둘 다 없음 → 완벽히 일치
    return 2.0 * intersection / union

def iou_score(pred, target, class_idx):
    pred_bin = (pred == class_idx)
    target_bin = (target == class_idx)
    intersection = (pred_bin & target_bin).sum().item()
    union = (pred_bin | target_bin).sum().item()
    if union == 0:
        return 1.0
    return intersection / union

# dice_score() softmax 제거
def dice_score_v2(pred_soft, target, epsilon=1e-6, exclude_class_0=True):
    """
    pred_soft: softmax 적용된 확률값 [B, C, D, H, W]
    """
    num_classes = pred_soft.shape[1]
    total_dice = 0
    classes = range(1, num_classes) if exclude_class_0 else range(num_classes)

    for cls in classes:
        pred_cls = pred_soft[:, cls]
        target_cls = (target == cls).float()

        intersect = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersect + epsilon) / (union + epsilon)
        total_dice += dice

    return total_dice / len(classes)

def multiclass_dice(pred, target, num_classes, exclude_class_0=True, epsilon=1e-6):
    """
    Multi-class Dice Score
    pred, target: shape (B, D, H, W) with integer class labels
    """
    classes = range(1, num_classes) if exclude_class_0 else range(num_classes)
    total_dice = 0.0

    for cls in classes:
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersect = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)
        dice = (2. * intersect + epsilon) / (union + epsilon)
        total_dice += dice

    return total_dice / len(classes)



# Argmax 기반 Hard Dice
def classwise_dice(pred, target, num_classes=5):
    pred_soft = F.softmax(pred, dim=1)
    dice_per_class = []

    for cls in range(num_classes):
        pred_cls = pred_soft[:, cls]
        target_cls = (target == cls).float()

        intersect = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)
        dice = (2. * intersect + 1e-6) / (union + 1e-6)
        dice_per_class.append(dice.item())

    return dice_per_class

"""
🔚 결론
훈련 중 loss 계산 → classwise_dice와 같은 soft Dice 기반이 좋음.

정확한 성능 평가 (inference 후 성능 비교 등) → per_class_dice를 사용하는 게 정확해.
"""

def per_class_dice(preds, masks, num_classes=5):
    """
    preds: [B, C, D, H, W] (logits)
    masks: [B, D, H, W] (ground truth)
    """
    pred_labels = torch.argmax(preds, dim=1)  # [B, D, H, W]
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (pred_labels == cls).float()
        gt_cls = (masks == cls).float()

        intersection = (pred_cls * gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum()
        dice = (2. * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())

    return dice_scores




import torch
from tqdm import tqdm
from monai.transforms import Resize

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = len(dataloader)

    loop = tqdm(dataloader, desc="🧪 Eval", leave=True)

    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            preds, _ = model(images)

            # sample-wise Resize (because monai.transforms.Resize doesn't support 5D directly)
            target_shape = list(masks.shape[-3:])  # (D, H, W)

            resize_fn = Resize(spatial_size=target_shape, mode="trilinear")
            preds_upsampled = torch.stack([
                resize_fn(p) for p in preds  # p: [C, D, H, W]
            ])

            # preds_upsampled = torch.stack([resize_fn(p.unsqueeze(0)).squeeze(0) for p in preds])

            preds_soft = torch.softmax(preds_upsampled, dim=1)  #> main의 Crossentropy에서 softmax가 진행된 채로 전달된다.
            loss = criterion(preds_upsampled, masks) #는 softmax 값 그대로 사용
            dice_v2 = dice_score_v2(preds_soft, masks) #는 cls값 사용

            total_loss += loss.item()
            total_dice += dice_v2.item()
            loop.set_postfix(loss=loss.item(), dice_v2=dice_v2.item())

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice

#================================================
import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

def eval_saved_predictions(pred_dir, gt_dir, num_classes=5):
    """
    저장된 prediction .nii.gz 파일과 GT를 비교하여 Dice / IoU 평가 및 성능 등급화.

    Args:
        pred_dir: 예측 결과 .nii.gz가 저장된 디렉터리
        gt_dir: ground truth가 포함된 BraTS directory
    """
    dice_all = []
    iou_all = []
    multiclass_dice_scores = []  # softmax 대신 argmax 기반 dice score

    # 성능 등급별 케이스 모음
    high_score_cases = []   # Dice >= 0.9
    middle_score_cases = [] # 0.7 <= Dice < 0.9
    low_score_cases = []    # Dice <= 0.5

    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if f.endswith("-pred_upscaled.nii.gz")
    ])

    for idx, fname in enumerate(tqdm(pred_files, desc="📊 Saved Evaluation")):
        case_id = fname.replace("-pred_upscaled.nii.gz", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, case_id, f"{case_id}-seg.nii.gz")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"❌ 파일 누락: {case_id}")
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.int64)
        gt = nib.load(gt_path).get_fdata().astype(np.int64)

        # 차원 정리
        pred = torch.tensor(pred).unsqueeze(0)  # [1, D, H, W]
        gt = torch.tensor(gt).unsqueeze(0)

        # class별 Dice/IoU
        dice_per_class = [dice_score(pred, gt, cls) for cls in range(num_classes)]
        iou_per_class = [iou_score(pred, gt, cls) for cls in range(num_classes)]
        dice_v2_score = multiclass_dice(pred, gt, num_classes)

        dice_all.append(dice_per_class)
        iou_all.append(iou_per_class)
        multiclass_dice_scores.append(dice_v2_score.item())

        # 등급 분류
        real_case_id = idx + 181  # 예: BraTS-PED-{case_id:05d}
        if dice_v2_score >= 0.9:
            high_score_cases.append((real_case_id, dice_v2_score.item()))
        elif dice_v2_score >= 0.7:
            middle_score_cases.append((real_case_id, dice_v2_score.item()))
        elif dice_v2_score < 0.51:
            low_score_cases.append((real_case_id, dice_v2_score.item()))

    # 평균 계산
    dice_all = np.array(dice_all)
    iou_all = np.array(iou_all)
    mean_dice = np.mean(dice_all, axis=0)
    mean_ious = np.mean(iou_all, axis=0)

    print("\n📌 평균 Dice per class:", [round(d, 4) for d in mean_dice])
    print("📌 평균 IoU  per class:", [round(i, 4) for i in mean_ious])
    print("📌 평균 Dice_v2 (argmax 기반):", np.mean(multiclass_dice_scores))

    # ✅ 성능 등급 출력
    print("\n✅ Dice 0.9 이상인 Case ID 목록:")
    print(high_score_cases)
    print("비율:", round(len(high_score_cases) / len(multiclass_dice_scores) * 100, 2), "%")

    print("\n✅ Dice 0.7 이상인 Case ID 비율:")
    print(middle_score_cases)
    print(round(len(middle_score_cases) / len(multiclass_dice_scores) * 100, 2), "%")

    print("\n❌ Dice 0.51 미만인 Case ID 비율:")
    print(low_score_cases)
    print(round(len(low_score_cases) / len(multiclass_dice_scores) * 100, 2), "%")


if __name__ == "__main__":
    pred_dir = "./predictions/aug_v2"
    gt_dir = "./data/BraTS-PEDs2024_Training"
    eval_saved_predictions(pred_dir, gt_dir)