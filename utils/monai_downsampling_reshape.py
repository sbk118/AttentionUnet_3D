import torch
from monai.transforms import Resize

# 리사이즈 트랜스폼 정의
resize_img = Resize(spatial_size=(128, 128, 32), mode="trilinear")

def monai_resize(images, is_mask=False):
    """
    이미지 또는 마스크를 리사이즈하는 함수
    - 이미지만 리사이즈하고, 마스크는 그대로 반환

    Args:
        images: np.ndarray
            - 이미지: shape (C, H, W, D)
            - 마스크: shape (H, W, D)
        is_mask: bool
            - True면 마스크로 간주 → 리사이즈 하지 않고 그대로 반환

    Returns:
        np.ndarray: 리사이즈된 이미지 또는 원본 마스크
    """
    if is_mask:
        # ❌ 마스크는 리사이즈하지 않고 그대로 반환
        return images
    else:
        # ✅ 이미지만 리사이즈 수행
        tensor = torch.tensor(images)  # [C, H, W, D]
        resized = resize_img(tensor).numpy()
        return resized
