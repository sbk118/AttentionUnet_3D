from train import *
from evaluation import *
from torch.utils.data import random_split
import os
from datasets.dataset import BraTSDataset
from torch.utils.data import DataLoader
from Attention_UNet_3D.segmentation.attention_unet_3d import UNet3D
import torch.nn as nn

# -----------------------------
# 🏁 메인 실행 함수
# -----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 사용 디바이스: {device}")

    # 하이퍼파라미터
    in_channels = 4
    out_channels = 5
    epochs = 2
    lr = 1e-4
    batch_size = 1

    # 데이터 준비
    case_list = sorted(os.listdir('./data/BraTS-PEDs2024_Training'))
    print(f"🔍 총 케이스 수: {len(case_list)}")
    dataset = BraTSDataset(data_dir='data/BraTS-PEDs2024_Training', case_list=case_list)

    # train, test split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    print(f"📊 Split: train={train_size}, val={val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 설정
    model = UNet3D(in_channels, out_channels, final_sigmoid=False, f_maps=32, layer_order='crg', num_groups=8)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습 루프
    print("🎯 학습 시작!")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, num_classes=5)
        print(f"[{epoch:02d}/{epochs}] 🎯 Train Loss: {train_loss:.4f} | 🧪 Val Loss: {val_loss:.4f} | 🎯 Dice: {val_dice:.4f}")


    # 모델 저장
    torch.save(model.state_dict(), "./models/attention_unet3d.pth")
    print("✅ 모델 저장 완료!")
    # torch.save(model.state_dict(), f"attention_unet3d_epoch{epoch}.pth")
    # print(f"📦 모델 저장 완료: attention_unet3d_epoch{epoch}.pth")


if __name__ == "__main__":
    main()