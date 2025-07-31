import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from utils.monai_downsampling_reshape import monai_resize

class BraTSDataset(Dataset):
    def __init__(self, data_dir, aug_dir, case_list, transform=None):
        self.data_dir = data_dir
        self.aug_dir = aug_dir
        self.case_list = case_list
        self.transform = transform
        self.modalities = ['t1c', 't1n', 't2f', 't2w']

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]


        # if case.startswith("augmented_"):
        #     image_path = os.path.join(self.aug_dir, f"{case}.nii")
        #     mask_path = os.path.join(self.aug_dir, f"{case.replace('resized_aug', 'seg-resized_aug')}.nii")
        #
        #     image = nib.load(image_path).get_fdata().astype(np.float32)  # (128, 128, 32) or (128, 128, 32, 1 or 4)
        #     # print(f"📐 [DEBUG] Augmented raw image shape: {image.shape}")
        #     image = np.transpose(image, (3, 0, 1, 2))
        #     # print(f"📐 [DEBUG] Augmented transpose image shape: {image.shape}")
        #     # 이미지 정규화
        #     image = (image - image.mean()) / (image.std() + 1e-8)
        #     image = torch.from_numpy(image).float()
        #     # print(f"📐 [DEBUG] Augmented torch image shape: {image.shape}")
        #
        #     mask = nib.load(mask_path).get_fdata().astype(np.int64)
        #     # print(f"📐 [DEBUG] Augmented raw mask shape: {mask.shape}") #불러온 mask
        #     mask = torch.tensor(mask).unsqueeze(0).squeeze(axis=4)
        #     # print(f"shape: {mask.shape}") #mask shape
        #
        #     # 마스크 업샘플링: [1, D, H, W] → [D, H, W]
        #     mask = monai_resize(mask[np.newaxis, ...], is_mask=True)[0]
        #     mask = mask.squeeze().numpy()

        if "_aug" in case:  # augmentation 데이터
            new_case, aug_num = case.split("_aug")
            num = aug_num.split("_")[0]
            img_file = os.path.join(self.aug_dir, f"{new_case}_aug{num}_img.nii.gz")
            image = nib.load(img_file).get_fdata().astype(np.float32)  # [128,128,32,4]
            image = torch.from_numpy(image).permute(0, 1, 2, 3).float()  # [4, D, H, W]
            image = (image - image.mean()) / (image.std() + 1e-8)
            # image = torch.from_numpy(image).float()

            # --- seg ----
            seg_path = os.path.join(self.aug_dir, f"{new_case}_aug{num}_seg.nii.gz")
            mask = nib.load(seg_path).get_fdata().astype(np.int64)
            mask = np.clip(mask, 0, 4)
            mask = torch.from_numpy(mask).long()
            mask = mask.squeeze()


        else:  # 원본 케이스
            case_path = os.path.join(self.data_dir, case)
            images = []
            for mod in self.modalities:
                img_path = os.path.join(case_path, f'{case}-{mod}.nii.gz')
                img = nib.load(img_path).get_fdata().astype(np.float32)
                images.append(img)
            images = np.stack(images, axis=0)  # [4, H, W, D]
            # print(f"📐 [DEBUG] Original stacked image shape: {images.shape}")
            images = monai_resize(images, is_mask=False)

            seg_path = os.path.join(case_path, f'{case}-seg.nii.gz')
            mask = nib.load(seg_path).get_fdata().astype(np.int64)
            mask = np.clip(mask, 0, 4)

            images = (images - images.mean()) / (images.std() + 1e-8)
            image = torch.from_numpy(images).float()
            mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask



# -------------------------
# 테스트용 코드
# -------------------------
if __name__ == "__main__":
    data_dir = "../data/BraTS-PEDs2024_Training"
    aug_dir = "../data/aug_v2"  # 증강 데이터 경로 지정
    #
    base_cases = sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith(".")
    ])[:3]  # 일부만 테스트

    # aug_cases = []
    # for case in base_cases:
    #     aug_cases.extend([f"augmented_{case.rsplit('-', 1)[0]}-resized_aug{i}" for i in range(5)])
    aug_cases = sorted([
        name for name in os.listdir(aug_dir)
        if os.path.isdir(os.path.join(aug_dir, name)) and not name.startswith(".")
    ])[:3]


    full_case_list = base_cases + aug_cases
    dataset = BraTSDataset(data_dir=data_dir, aug_dir=aug_dir, case_list=full_case_list)

    for i in range(3):
        img, mask = dataset[i]
        print(f"✅ Sample {i}: image shape {img.shape}, mask shape {mask.shape}, unique labels: {torch.unique(mask)}")

    # # 직접 테스트
    # aug_case = "BraTS-PED-00001-000_aug0_img"
    # dataset = BraTSDataset(data_dir=data_dir, aug_dir=aug_dir, case_list=[aug_case])
    # img, mask = dataset[0]
    # print(f"✅ image shape: {img.shape}, mask shape: {mask.shape}")