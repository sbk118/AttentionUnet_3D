from tqdm import tqdm
import torch
from utils.metrics import dice_coefficient  # 선택

def evaluate(model, dataloader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0
    all_dice = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="🔵 Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            _, mean_dice = dice_coefficient(preds, masks, num_classes)
            all_dice.append(mean_dice)

    avg_loss = total_loss / len(dataloader)
    avg_dice = sum(all_dice) / len(all_dice)
    return avg_loss, avg_dice
