import nibabel as nib
import numpy as np
import pyvista as pv
import os
from math import cos, sin, radians
import tensorflow as tf
from utils.resize import resize_xy_and_crop_z

# --- GPU 확인 ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU {gpus[0].name} 사용 가능")
else:
    print("❌ GPU 사용 불가 (현재 CPU로 실행 중)")

# --- Step 1: 파일 경로 설정 ---
t1_path = "../data/testing/BraTS-PED-00006-000/BraTS-PED-00006-000-t2f.nii.gz"
seg_path = "../predictions/BraTS-PED-00006-000-pred.nii.gz"

# --- Step 2: 데이터 로드 ---
t1_img = nib.load(t1_path).get_fdata()
seg_img = nib.load(seg_path).get_fdata()

# 원본 crop
t2f_cropped = resize_xy_and_crop_z(t1_img, target_xy=128, target_z=32)  # ✅ (128, 128, 32)

# --- Step 3: 정규화 및 마스크 처리 ---
brain = (t2f_cropped - np.min(t2f_cropped)) / (np.max(t2f_cropped) - np.min(t2f_cropped))
tumor_mask = (seg_img > 0).astype(np.uint8)

# --- Step 4: PyVista surface 추출 ---
brain_surface = pv.wrap(brain).contour([0.3])
tumor_surface = pv.wrap(tumor_mask).contour([0.5])

# --- Step 5: Plotter 설정 ---
plotter = pv.Plotter(off_screen=True)
focal_point = np.array(brain_surface.center)

# --- Step 5.1: 뇌 앞쪽(+Y)에만 화살표 추가 ---
# --- Step 5.1: 뇌 앞쪽(+Z)에만 화살표 추가 (전면) ---
arrow_offset = np.array([0, 0, 100])  # +Z 방향으로 이동
arrow_start = focal_point + arrow_offset
arrow_direction = [0, 0, 1]  # +Z가 front

arrow = pv.Arrow(
    start=arrow_start,
    direction=arrow_direction,
    tip_length=0.4,
    tip_radius=3,
    shaft_radius=1.5
)
plotter.add_mesh(arrow, color='blue', name='front_arrow')



# --- Step 5.2: 뇌와 종양 surface 추가 ---
plotter.add_mesh(brain_surface, color="ivory", opacity=0.2)
plotter.add_mesh(tumor_surface, color="red", opacity=1.0)
plotter.add_axes()

# --- Step 6: 회전 중심 및 반지름 설정 ---
radius = 400

# --- Step 7: 동영상 저장 ---
output_filename = "../video/brain_tumor_rotationPED06_yaxis.mp4"
plotter.open_movie(output_filename, framerate=17, codec="libx264")

n_frames = 90
for i in range(n_frames):
    angle = radians(i * 4)
    x = radius * cos(angle)
    z = radius * sin(angle)
    position = focal_point + np.array([x, 0, z])  # ✅ Y축 기준으로 회전
    plotter.camera_position = [position, focal_point, [0, 1, 0]]  # up-vector도 Y로 설정
    plotter.render()
    plotter.write_frame()

plotter.close()
print(f"✅ Y축 기준 회전 영상 저장 완료: {output_filename}")
