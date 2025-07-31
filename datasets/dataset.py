import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from utils.resize import resize_xy_and_crop_z

class BraTSDataset(Dataset):
    def __init__(self, data_dir, case_list, transform=None):
        """
        Args:
            data_dir (str): 환자 케이스 폴더들이 있는 상위 디렉토리 경로
            case_list (list): 사용할 환자 폴더 이름 리스트
            transform (callable, optional): (image, label) -> (image, label)
        """
        self.data_dir = data_dir
        self.case_list = case_list
        self.transform = transform
        self.modalities = ['t1c', 't1n', 't2f', 't2w']  # 4가지 MRI 모달리티

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        case_path = os.path.join(self.data_dir, case)

        # ========= 4채널 이미지 로딩 =========
        images = []
        for modality in self.modalities:
            img_path = os.path.join(case_path, f'{case}-{modality}.nii.gz')
            img = nib.load(img_path).get_fdata().astype(np.float32)  # shape: [H, W, D]
            images.append(img)
        images = np.stack(images, axis=0)  # shape: [4, H, W, D]

        # ========= 세그멘테이션 마스크 로딩 =========
        seg_path = os.path.join(case_path, f'{case}-seg.nii.gz')
        seg = nib.load(seg_path).get_fdata().astype(np.int64)  # shape: [H, W, D]
        seg = np.clip(seg, 0, 4)  # 클래스 레이블은 0~4만 허용

        # # resize
        # images = np.stack(images, axis=0)  # [4, H, W, D]
        # images = resize_xy_and_crop_z(images, target_xy=128, target_z=32, order=3)
        #
        # seg = resize_xy_and_crop_z(seg, target_xy=128, target_z=32, order=0)  # mask는 order=0 (nearest)

        # ========= intensity 정규화 =========
        images = (images - images.mean()) / (images.std() + 1e-8)

        # ========= numpy → torch tensor =========
        images = torch.from_numpy(images).float()  # shape: [4, H, W, D]
        seg = torch.from_numpy(seg).long()         # shape: [H, W, D]

        # ========= transform 적용 (선택) =========
        if self.transform:
            images, seg = self.transform(images, seg)

        return images, seg
