from scipy.ndimage import zoom
import numpy as np

def resize_xy_and_crop_z(volume, target_xy=128, target_z=32, order=3):
    """
    XY는 resize, Z는 가운데 crop
    Args:
        volume: shape (C, H, W, D) or (H, W, D)
        order: 3 for image, 0 for mask
    Returns:
        resized_volume: shape (..., target_xy, target_xy, target_z)
    """
    is_4d = volume.ndim == 4
    if is_4d:
        c, h, w, d = volume.shape
        zoom_factors = [1.0, target_xy / h, target_xy / w, 1.0]
    else:
        h, w, d = volume.shape
        zoom_factors = [target_xy / h, target_xy / w, 1.0]

    resized = zoom(volume, zoom=zoom_factors, order=order)

    # 가운데 Z 32장 슬라이싱
    z_start = (resized.shape[-1] - target_z) // 2
    z_end = z_start + target_z
    cropped = resized[..., z_start:z_end]

    return cropped.astype(np.float32)
