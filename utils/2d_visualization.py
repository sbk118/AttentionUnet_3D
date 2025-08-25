import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

base_path = r"../data/BraTS-PEDs2024_Training/BraTS-PED-00181-000"
num_samples = 5  # 보고 싶은 케이스 개수

modalities = ['t1c', 'seg', 'pred']

all_cases = sorted([d for d in os.listdir(base_path) if d.startswith("BraTS-PED")])
# one_case = ["BraTS-PED-00182-000"]

def normalize_img(img):
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

colors = {
    1: [1, 0, 0, 0.6],   # 빨강
    2: [1.0, 0.65, 0.0, 0.6], #오렌지
    3: [1, 1, 0, 0.6],   # 노랑
    4: [0, 1, 0, 0.6]   # 초록
}



class_names = {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED"
}

# 시각화용 패치 생성
patches = [
    mpatches.Patch(color=colors[cls], label=f"{cls}: {class_names[cls]}")
    for cls in colors
]

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')

legend = ax.legend(handles=patches, loc='center', ncol=2, frameon=True)
legend.get_frame().set_facecolor('lightgray')  # 범례 박스 배경색 설정
legend.get_frame().set_edgecolor('black')     # 범례 박스 테두리 색

plt.tight_layout()
plt.show()



for i, case in enumerate(all_cases[1:num_samples]):
# for i, case in enumerate(one_case):
    t1c_path = os.path.join(base_path, case, f"{case}-t1c.nii.gz")
    seg_path = os.path.join(base_path, case, f"{case}-seg.nii.gz")
    pred_path = os.path.join(base_path, case, f"{case}-pred.nii.gz")

    t1c_img = nib.load(t1c_path).get_fdata()
    seg_img = nib.load(seg_path).get_fdata() if os.path.exists(seg_path) else None
    pred_img = nib.load(pred_path).get_fdata() if os.path.exists(pred_path) else None

    if seg_img is not None:
        lesion_counts = [(pred_img[:, :, idx] != 0).sum() for idx in range(pred_img.shape[2])]
        slice_idx = int(np.argmax(lesion_counts))
    else:
        slice_idx = t1c_img.shape[2] // 2

    t1c_slice = t1c_img[:, :, slice_idx]
    t1c_norm = normalize_img(t1c_slice.T)

    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))

    # t1c
    axes[0].imshow(t1c_norm, cmap='gray', origin='lower')
    axes[0].set_title(f"{case}\nt1c")
    axes[0].axis('off')

    # seg
    if seg_img is not None:
        seg_slice = seg_img[:, :, slice_idx]
        rgba_mask = np.zeros((*seg_slice.T.shape, 4))
        for cls, color in colors.items():
            mask = seg_slice.T == cls
            rgba_mask[mask] = color

        axes[1].imshow(t1c_norm, cmap='gray', origin='lower')
        axes[1].imshow(rgba_mask, origin='lower')
        axes[1].set_title("Ground Truth (seg)")
    else:
        axes[1].text(0.5, 0.5, 'No seg data', ha='center', va='center')
        axes[1].set_facecolor('lightgray')
    axes[1].axis('off')

    # pred
    if pred_img is not None:
        pred_slice = pred_img[:, :, slice_idx]
        rgba_mask_pred = np.zeros((*pred_slice.T.shape, 4))
        for cls, color in colors.items():
            mask = pred_slice.T == cls
            rgba_mask_pred[mask] = color

        axes[2].imshow(t1c_norm, cmap='gray', origin='lower')
        axes[2].imshow(rgba_mask_pred, origin='lower')
        axes[2].set_title("Model Prediction (pred)")
    else:
        axes[2].text(0.5, 0.5, 'No pred data', ha='center', va='center')
        axes[2].set_facecolor('lightgray')
    axes[2].axis('off')

    # 하단 여백 확보
    plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()
    plt.show()  # 여기서 창 닫으면 다음 케이스 표시됨