import os
import numpy as np
import torch
import torchio as tio

data_dir = r'../data/BraTS-PEDs2024_Training'
save_dir = r'../data/Augmentation'
os.makedirs(save_dir, exist_ok=True)

tfm = tio.Compose([
    tio.ZNormalization(),
    tio.RandomFlip(axes=('LR', 'AP', 'IS')),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    tio.RandomElasticDeformation(num_control_points=5, max_displacement=4, locked_borders=2),
    tio.RandomNoise(mean=0, std=0.05),
    tio.RandomBlur(std=(0.1, 0.5), p=0.3),
    tio.RandomGamma(log_gamma=(-0.2, 0.2)),
    tio.RandomBiasField(),
])
n_aug = 3

def augment_and_save(patient_id, img_path, seg_path):
    img_np = np.load(img_path).astype(np.float32)
    seg_np = np.load(seg_path).astype(np.int16)
    img_tensor = torch.from_numpy(img_np).permute(3, 0, 1, 2)
    if seg_np.ndim == 4:
        seg_tensor = torch.from_numpy(seg_np).squeeze(-1).unsqueeze(0).long()
    elif seg_np.ndim == 3:
        seg_tensor = torch.from_numpy(seg_np).unsqueeze(0).long()
    else:
        raise ValueError(f"{patient_id} seg shape 이상: {seg_np.shape}")
    image = tio.ScalarImage(tensor=img_tensor)
    mask = tio.LabelMap(tensor=seg_tensor)
    subject = tio.Subject(image=image, mask=mask)
    for i in range(n_aug):
        aug_subject = tfm(subject)
        img_aug = aug_subject['image'].data
        mask_aug = aug_subject['mask'].data
        img_save_path = os.path.join(save_dir, f"{patient_id}_aug{i}_img.pt")
        mask_save_path = os.path.join(save_dir, f"{patient_id}_aug{i}_seg.pt")
        torch.save(img_aug.clone(), img_save_path)
        torch.save(mask_aug.clone(), mask_save_path)
        print(f"{patient_id} 증강 {i} 저장 완료")

for idx in range(1, 267):
    patient_id = f'BraTS-PED-{idx:05d}-000'
    img_path = os.path.join(data_dir, f'{patient_id}_4d.npy')
    seg_path = os.path.join(data_dir, f'{patient_id}_seg.npy')
    if not os.path.exists(img_path):
        print(f'{patient_id} 4d 파일 없음, 건너뜀')
        continue
    if not os.path.exists(seg_path):
        print(f'{patient_id} 세그멘테이션 파일 없음, 건너뜀')
        continue
    try:
        augment_and_save(patient_id, img_path, seg_path)
    except Exception as e:
        print(f'{patient_id} 증강 실패: {str(e)}')

print('전체 증강 및 저장 완료')
