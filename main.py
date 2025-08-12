from datasets.dataset_v3 import BraTSDataset
from torch.utils.data import DataLoader, random_split
from Attention_UNet_3D.segmentation.attention_unet_3d import UNet3D
from train import train
from eval import evaluate
import torch
import torch.nn as nn
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {device}")

    # hyperparameters
    in_channels = 4
    out_channels = 5
    epochs = 100
    lr = 1e-4
    batch_size = 1
    patience = 20  # Early stopping patience

    data_path = './data/BraTS-PEDs2024_Training'
    # 원본 케이스
    base_cases = sorted([
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name)) and not name.startswith(".")
    ])

    #증강 케이스
    aug_path = "./data/aug_v2"
    # aug_cases = []
    # for case in base_cases[:175]:  # 학습용만
    #     aug_cases.extend([
    #         f"augmented_{case.rsplit('-', 1)[0]}-resized_aug{i}" for i in range(5)
    #     ])

    aug_cases = sorted([
        fname.replace("_img.nii.gz", "")
        for fname in os.listdir(aug_path)
        if fname.endswith("_img.nii.gz")
    ])

    train_val_set = base_cases[:175] + aug_cases
    # test_set = base_cases[175:]
    # print(f"🔍 총 케이스 수(원본): {len(train_val_set)}")
    print(f"🔍 총 케이스 수 (원본 + 증강): {len(train_val_set)}")
    #'data/augmentation/augmented_BraTS-PED-00110-000-resized_aug3.nii'
    #'data/augmentation/augmented_BraTS-PED-00110-resized_aug3.nii'

    # dataset = BraTSDataset(data_path, train_val_set)
    dataset = BraTSDataset(data_dir=data_path, aug_dir=aug_path, case_list=train_val_set)

    train_len = int(len(dataset) * 0.7)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    print(f"📊 Split: train={train_len}, val={val_len}")

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    model = UNet3D(in_channels, out_channels, final_sigmoid=False, f_maps=32)
    model.to(device)

    # 제안2: weight를 다르게
    weights = torch.tensor([0.01, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch}] 🎯 Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | 🧪 Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./models/best_attention_unet3d_add_aug_v2.pth")
            print("📦 Best model saved!")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    print("✅ 학습 완료!")

if __name__ == '__main__':
    main()
