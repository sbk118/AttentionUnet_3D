import torch
import torch.nn.functional as F
from tqdm import tqdm
from monai.transforms import Resize
from eval_v2 import dice_score_v2

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0
    total_dice_v2 = 0 #원래 내 dice score 계산 값
    num_batches = len(dataloader)

    loop = tqdm(dataloader, desc="🚂 Train", leave=True)

    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds, _ = model(images)  # preds: [B, C, 128, 128, 32]

        # ✅ Resize (1회만 정의하여 반복문에서 재사용) → 속도 개선
        target_spatial = list(masks.shape[-3:])  # [D, H, W]
        resize_fn = Resize(spatial_size=target_spatial, mode="trilinear")

        # ⚠️ 각 배치 sample별 Resize 적용 (MONAI는 [C, D, H, W]만 지원)
        preds_upsampled = torch.stack([resize_fn(p) for p in preds])  # [B, C, 240, 240, 155]

        # 🎯 Loss 계산 (CrossEntropy 기준: logits 사용 → softmax X)
        loss = criterion(preds_upsampled, masks)
        loss.backward()
        optimizer.step()

        # 🎯 Dice 계산 (softmax 포함된 함수 사용)
        preds_soft = torch.softmax(preds_upsampled, dim=1)
        dice_v2 = dice_score_v2(preds_upsampled, masks)
        # dice_v2 = multiclass_dice(preds_soft, masks)

        total_loss += loss.item()
        total_dice_v2 += dice_v2.item()
        loop.set_postfix(loss=loss.item(), dice_v2=dice_v2.item())

    avg_loss = total_loss / num_batches
    # avg_dice = total_dice / num_batches
    avg_dice_v2 = total_dice_v2 / num_batches
    return avg_loss, avg_dice_v2
