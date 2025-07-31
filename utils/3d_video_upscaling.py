import os
import numpy as np
import nibabel as nib
import pyvista as pv
from math import cos, sin, radians
import tensorflow as tf

# --------------------
# GPU 확인
# --------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU {gpus[0].name} 사용 가능")
else:
    print("❌ GPU 사용 불가 (현재 CPU로 실행 중)")

# --------------------
# 경로 설정
# --------------------
data_dir = "../data/BraTS-PEDs2024_Training"
pred_dir = "../predictions/aug_v2"
video_dir = "../video/aug_v2"
os.makedirs(video_dir, exist_ok=True)

# --------------------
# 클래스별 색상 정의
# --------------------
label_colors = {
    1: "red",
    2: "orange",
    3: "yellow",
    4: "green"
}

# --------------------
# 회전 영상 생성 함수
# --------------------
def make_rotation_video(filename, surfaces, colors, opacities, focal_point):
    plotter = pv.Plotter(off_screen=True)

    # 방향 화살표
    arrow = pv.Arrow(
        start=focal_point + np.array([0, 0, 100]),
        direction=[0, 0, 1],
        tip_length=0.4,
        tip_radius=3,
        shaft_radius=1.5
    )
    plotter.add_mesh(arrow, color='blue', name='front_arrow')

    for surface, color, opacity in zip(surfaces, colors, opacities):
        plotter.add_mesh(surface, color=color, opacity=opacity)

    plotter.add_axes()
    plotter.open_movie(filename, framerate=17, codec="libx264")

    for i in range(90):
        angle = radians(i * 4)
        x = 400 * cos(angle)
        z = 400 * sin(angle)
        camera_pos = np.array([x, 0, z]) + focal_point
        plotter.camera_position = [camera_pos.tolist(), focal_point.tolist(), [0, 1, 0]]
        plotter.render()
        plotter.write_frame()

    plotter.close()

# --------------------
# 케이스 순회
# --------------------
case_list = sorted([f for f in os.listdir(pred_dir) if f.endswith("-pred_upscaled.nii.gz")])
print("case_list:", case_list)
for pred_file in case_list:
    case_id = pred_file.replace("-pred_upscaled.nii.gz", "")
    print(f"\n🔀 예측 영상 제작: {case_id}")

    t2f_path = os.path.join(data_dir, case_id, f"{case_id}-t2f.nii.gz")
    gt_path = os.path.join(data_dir, case_id, f"{case_id}-seg.nii.gz")
    pred_path = os.path.join(pred_dir, pred_file)

    gt_output_path = os.path.join(video_dir, f"{case_id}_gt.mp4")
    pred_output_path = os.path.join(video_dir, f"{case_id}_pred.mp4")

    try:
        t2f_data = nib.load(t2f_path).get_fdata()
        gt_data = nib.load(gt_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()
    except Exception as e:
        print(f"❌ {case_id} 로드 오류: {e}")
        continue

    brain = (t2f_data - np.min(t2f_data)) / (np.max(t2f_data) - np.min(t2f_data))
    brain_surface = pv.wrap(brain).contour([0.3])
    focal_point = np.array(brain_surface.center)

    # GT surface
    gt_surfaces, gt_colors, gt_opacities = [brain_surface], ["ivory"], [0.2]
    for label, color in label_colors.items():
        mask = (gt_data == label).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        surface = pv.wrap(mask).contour([0.5])
        gt_surfaces.append(surface)
        gt_colors.append(color)
        gt_opacities.append(1.0)

    # Pred surface
    pred_surfaces, pred_colors, pred_opacities = [brain_surface], ["ivory"], [0.2]
    for label, color in label_colors.items():
        mask = (pred_data == label).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        surface = pv.wrap(mask).contour([0.5])
        pred_surfaces.append(surface)
        pred_colors.append(color)
        pred_opacities.append(0.6)

    # 저장
    make_rotation_video(gt_output_path, gt_surfaces, gt_colors, gt_opacities, focal_point)
    print(f"✅ GT 저장 완료: {gt_output_path}")

    make_rotation_video(pred_output_path, pred_surfaces, pred_colors, pred_opacities, focal_point)
    print(f"✅ 예측 저장 완료: {pred_output_path}")
