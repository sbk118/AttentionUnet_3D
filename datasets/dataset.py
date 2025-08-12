import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from utils.monai_downsampling_reshape import monai_resize


class BraTSDataset(Dataset):
    def __init__(self, data_dir, case_list, transform=None):
        self.data_dir = data_dir
        self.case_list = case_list
        self.transform = transform
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        # self.aug_num = [0,1,2,3,4]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        case_path = os.path.join(self.data_dir, case)

        # --- 이미지 로딩 및 MONAI 리사이즈 ---
        images = []
        for mod in self.modalities:
            img = nib.load(os.path.join(case_path, f'{case}-{mod}.nii.gz')).get_fdata().astype(np.float32)
            images.append(img)
        images = np.stack(images, axis=0)  # [4, H, W, D]
        images = monai_resize(images, is_mask=False)  # [4, 128, 128, 32]4

        # --- 정규화 ---
        images = (images - images.mean()) / (images.std() + 1e-8)

        # --- 변환 ---
        images = torch.from_numpy(images).float()

        # --- 마스크 로딩 (GT는 원본 해상도 유지) ---
        seg = nib.load(os.path.join(case_path, f'{case}-seg.nii.gz')).get_fdata().astype(np.int64)
        seg = np.clip(seg, 0, 4)

        # --- 변환 ---
        seg = torch.from_numpy(seg).long()

        if self.transform:
            images, seg = self.transform(images, seg)

        return images, seg

import numpy as np
import torch
from collections import Counter

def compute_class_distribution(dataset, max_samples=20):
    label_counter = Counter()
    total_voxels = 0

    for i in range(min(len(dataset), max_samples)):
        _, seg = dataset[i]  # seg: torch.Tensor [H, W, D]
        labels, counts = torch.unique(seg, return_counts=True)

        for label, count in zip(labels.tolist(), counts.tolist()):
            label_counter[int(label)] += count
            total_voxels += count

    print("📊 클래스별 비율 (상위 {}개 샘플 기준):".format(max_samples))
    for label in sorted(label_counter.keys()):
        ratio = label_counter[label] / total_voxels * 100
        print(f"  🔹 클래스 {label}: {label_counter[label]:,} voxels ({ratio:.2f}%)")

# ==========================
# 🔍 직접 실행할 경우 테스트 코드
# ==========================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = "../data/BraTS-PEDs2024_Training"
    case_list = sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith(".")
    ])  # 상위 10개 케이스만 확인
    samples = len(case_list)
    dataset = BraTSDataset(data_dir, case_list)

    compute_class_distribution(dataset, max_samples=samples)  # 20개만 확인
    all_labels = []
    for i in range(len(dataset)):
        img, seg = dataset[i]
        # print(i,"번째")
    #     print(f"✅ image shape: {img.shape}, dtype: {img.dtype}, range: ({img.min():.2f}, {img.max():.2f})")
    #     print(f"✅ mask shape: {seg.shape}, unique values: {torch.unique(seg)}")
        all_labels.append(torch.unique(seg))

    flat = torch.cat(all_labels)
    print("\n💡 전체 라벨 분포 (상위 10개 기준):", torch.unique(flat))
