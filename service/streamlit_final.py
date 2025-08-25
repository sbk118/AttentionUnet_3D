### cd "C:\Users\KDT_32\PycharmProjects\MiniProject\1\seoyoung\mini_2"
### streamlit run streamlit_final.py

# === [공통 라이브러리 및 설정] ===
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pandas as pd
import numpy as np
import glob
import json
import ast
import torch
from datetime import datetime
# from skimage.transform import resize
import nibabel as nib
import matplotlib.pyplot as plt

from login import verify_user
from config import NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, TRAIN_CUDA
from transforms import train_transform, train_transform_cuda, val_transform, val_transform_cuda
from unet3d import UNet3D
from monai.transforms import Resize
from infer_video import make_video
import pyvista as pv
from math import cos, sin, radians

st.set_page_config(page_title="의료 대시보드", layout="wide")

# ✔️ 세션 상태 초기화
for key in ['logged_in', 'doctor_id', 'user_id', 'logout', 'selected_patient_id', 'search']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['logged_in', 'logout'] else None if key != 'search' else ""

MEMO_DIR = "patient_memos"
os.makedirs(MEMO_DIR, exist_ok=True)

def get_memo_file_path(patient_id):
    return os.path.join(MEMO_DIR, f"{patient_id}_memo.json")

def load_memo(patient_id):
    file_path = get_memo_file_path(patient_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memo_data = json.load(f)
                return memo_data.get('content', '')
        except json.JSONDecodeError:
            return ""
    return ""

def save_memo(patient_id, memo_content):
    file_path = get_memo_file_path(patient_id)
    memo_data = {
        'patient_id': patient_id,
        'content': memo_content,
        'last_updated': datetime.now().isoformat()
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(memo_data, f, ensure_ascii=False, indent=4)

@st.cache_data(show_spinner=False)
def load_patient_data():
    df = pd.read_csv("patient_data.csv")
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")

    if "read_status" not in df.columns:
        df["read_status"] = df["reading_date"].notna()

    if "mri_image_path" not in df.columns:
        df["mri_image_path"] = df["patient_id"].apply(
            lambda x: sorted(glob.glob(f"images/{x}_*.png")) or [f"images/{x}.png"]
        )
    else:
        def parse_image_path(path_str):
            if isinstance(path_str, str) and path_str.startswith("[") and path_str.endswith("]"):
                try:
                    return ast.literal_eval(path_str)
                except:
                    return [path_str]
            elif isinstance(path_str, str) and path_str.strip():
                return [path_str]
            return []

        df["mri_image_path"] = df["mri_image_path"].apply(parse_image_path)

        for idx, row in df.iterrows():
            if not isinstance(row['mri_image_path'], list) or not row['mri_image_path']:
                found_paths = sorted(glob.glob(f"images/{row['patient_id']}_*.png"))
                if not found_paths:
                    default_path = f"images/{row['patient_id']}.png"
                    found_paths = [default_path] if os.path.exists(default_path) else []
                df.at[idx, 'mri_image_path'] = found_paths

    if "mri_video_path" not in df.columns:
        df["mri_video_path"] = df["patient_id"].apply(lambda x: f"videos/{x}.mp4")
    if "doctor_name" not in df.columns:
        df["doctor_name"] = ""
    if "reading_date" not in df.columns:
        df["reading_date"] = pd.NaT
    if "followup_date" not in df.columns:
        df["followup_date"] = pd.NaT

    df["followup_date"] = pd.to_datetime(df["followup_date"], errors="coerce")
    df["reading_date"] = pd.to_datetime(df["reading_date"], errors="coerce")

    return df

# ========================
# 판독 완료 목록 표시
# ========================
def generate_3d_video(case_id, seg_img, label_type: str, label_discri: bool, output_dir):
    t1_path = rf"./data/BraTS-PED-{case_id}-000/BraTS-PED-{case_id}-000-t1c.nii.gz"
    if not os.path.exists(t1_path):
        print(f"❌ T1 image not found: {t1_path}")
        return

    t1_img = nib.load(t1_path).get_fdata().astype(np.float32)
    brain = (t1_img - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img) + 1e-8)
    tumor_mask = (seg_img > 0).astype(np.uint8)
    label1_mask = (seg_img == 1).astype(np.uint8)
    label2_mask = (seg_img == 2).astype(np.uint8)
    label3_mask = (seg_img == 3).astype(np.uint8)
    label4_mask = (seg_img == 4).astype(np.uint8)

    brain_surface = pv.wrap(brain).contour([0.3])
    tumor_surface = pv.wrap(tumor_mask).contour([0.5])
    l1_surf = pv.wrap(label1_mask).contour([0.5])
    l2_surf = pv.wrap(label2_mask).contour([0.5])
    l3_surf = pv.wrap(label3_mask).contour([0.5])
    l4_surf = pv.wrap(label4_mask).contour([0.5])

    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(off_screen=True)
    focal_point = np.array(brain_surface.center)

    arrow = pv.Arrow(start=focal_point + [0, 0, 100], direction=[0, 0, 1], tip_length=0.4, tip_radius=3, shaft_radius=1.5)
    plotter.add_mesh(arrow, color='blue', name='front_arrow')
    plotter.add_mesh(brain_surface, color="ivory", opacity=0.2)

    if label_discri:
        for s, c in zip([l1_surf, l2_surf, l3_surf, l4_surf], ["red", "orange", "yellow", "green"]):
            try:
                plotter.add_mesh(s, color=c, opacity=0.6)
            except:
                print(f"⚠️ No surface for label {c}")
    else:
        plotter.add_mesh(tumor_surface, color="red", opacity=1.0)

    plotter.add_axes()

    output_path = f"./output/video/3dvid_{case_id}_{label_type}_{label_discri}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plotter.open_movie(output_path, framerate=17, codec="libx264")

    radius = 400
    for i in range(90):
        angle = radians(i * 4)
        x = radius * cos(angle)
        z = radius * sin(angle)
        position = focal_point + np.array([x, 0, z])
        plotter.camera_position = [position, focal_point, [0, 1, 0]]
        plotter.render()
        plotter.write_frame()

    plotter.close()
    print(f"🎥 Saved video: {output_path}")

def make_video(case_num):
    pred_path = f"./output/pred/PED_{case_num}_pred.npy"
    if not os.path.exists(pred_path):
        print(f"❌ Prediction 파일이 없습니다: {pred_path}")
        return
    pred = np.load(pred_path)
    generate_3d_video(case_num, pred, "pred", True, "output/video")

def show_read_complete_section():
    st.subheader("📁 판독 완료된 MRI")
    df = load_patient_data()
    read_df = df[(df["read_status"] == True) & (df["doctor_id"].astype(str) == st.session_state.user_id)].sort_values(by="scan_date", ascending=False)

    for index, row in read_df.iterrows():
        with st.expander(f"🧐 {row['name']} | {row['scan_date'].date()}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                # case number 계산 (예시: patient_id의 숫자 부분을 추출해서 5자리로 맞춤)
                import re
                match = re.search(r'\d+', str(row['patient_id']))
                case_number = f"{int(match.group()):05d}" if match else "N/A"

                st.markdown(
                    f"""
                    **🆔 ID**: {row['patient_id']} &nbsp;&nbsp;
                    **🔢 case number**: {row['case_number']} &nbsp;&nbsp;
                    **👵 나이/성별**: {row['age']} / {row['gender']} &nbsp;&nbsp;
                    **📆 촬영일**: {row['scan_date'].date()} &nbsp;&nbsp;
                    **📆 판독일**: {row['reading_date'].date()} &nbsp;&nbsp;
                    **📆 예약일**: {row['followup_date'].date() if pd.notnull(row['followup_date']) else '미지정'} &nbsp;&nbsp;
                    **👨‍⚕️ 담당의**: {row['doctor_name'] if row['doctor_name'] else '미지정'} &nbsp;&nbsp;
                    """,
                    unsafe_allow_html=True
                )

                # 1. 종양 정보 준비
                if 'tumor_size(cm)' in row and not pd.isnull(row['tumor_size(cm)']):
                    tumor_info = f"📏 Volume: {row['tumor_size(cm)']} mm³\n\n"
                else:
                    tumor_info = "📏 Volume: 정보 없음\n\n"

                # 2. 메모만 불러오기
                memo_content = load_memo(row['patient_id'])
                default_text = tumor_info + memo_content

                # 3. text_area로 표시
                edited_text = st.text_area(" ", value=default_text, height=150, key=f"memo_{row['patient_id']}")

                # 4. 저장 시 종양 정보 제거
                if edited_text.startswith(tumor_info):
                    cleaned_memo = edited_text[len(tumor_info):]
                else:
                    cleaned_memo = edited_text

                if cleaned_memo != memo_content:
                    save_memo(row['patient_id'], cleaned_memo)
                    st.toast("메모가 저장되었습니다!", icon="📝")

            # ✅ UNet 예측 및 영상 생성 기능
            # st.markdown("### 🧠 3D UNet 뇌 종양 이미지 생성")
            case_input = st.number_input("Case Number", min_value=0, step=1, format="%d", key=f"case_input_{row['patient_id']}")
            CASE_ID = f"{case_input:05d}"

            if st.button("이미지 생성", key=f"predict_button_{row['patient_id']}"):
                MODEL_PATH = "epoch41.pth"
                INPUT_ROOT = "./data"
                OUTPUT_DIR = "output/pred"
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                with st.spinner("모델 추론 중..."):
                    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES + 1 if BACKGROUND_AS_CLASS else NUM_CLASSES)
                    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                    model = model.to(DEVICE).eval()

                    case_path = os.path.join(INPUT_ROOT, f"BraTS-PED-{CASE_ID}-000")
                    if not os.path.exists(case_path):
                        st.error(f"❌ 파일이 없습니다: {case_path}")
                        continue

                    transform = Resize(spatial_size=(128, 128, 32), mode="trilinear")
                    modality_list = ['t1n', 't1c', 't2w', 't2f']
                    modalities = []

                    for m in modality_list:
                        modality_path = os.path.join(case_path, f"BraTS-PED-{CASE_ID}-000-{m}.nii.gz")
                        nifti_img = nib.load(modality_path)
                        img_data = nifti_img.get_fdata()
                        resized_np = transform(torch.from_numpy(img_data).unsqueeze(0)).squeeze(0).numpy()
                        modalities.append(resized_np)

                    stacked_img = np.stack(modalities, axis=0)
                    data_dict = {'image': stacked_img}
                    processed_out = val_transform(data_dict)

                    with torch.no_grad():
                        output = model(processed_out["image"].unsqueeze(0).to(DEVICE))
                        pred = torch.argmax(output, dim=1)

                    pred = Resize(spatial_size=(240, 240, 155), mode="nearest")(pred).squeeze(0).cpu().numpy().astype(np.int8)
                    out_path = os.path.join(OUTPUT_DIR, f"PED_{CASE_ID}_pred.npy")
                    np.save(out_path, pred)
                    st.success(f"✅ 저장 완료: {out_path}")

                    t1c_img = nib.load(os.path.join(case_path, f"BraTS-PED-{CASE_ID}-000-t1c.nii.gz")).get_fdata()
                    lesion_counts = [(pred[:, :, idx] > 0).sum() for idx in range(pred.shape[2])]
                    default_idx = int(np.argmax(lesion_counts))
                    idx = st.slider("Z 축 슬라이스 번호", 0, pred.shape[2] - 1, default_idx, key=f"slider_{CASE_ID}")

                    t1c_slice = t1c_img[:, :, idx].T
                    pred_slice = pred[:, :, idx].T
                    t1c_norm = (t1c_slice - t1c_slice.min()) / (t1c_slice.max() - t1c_slice.min() + 1e-8)

                    colors = {1:[1,0,0,0.6], 2:[0,1,0,0.6], 3:[0,0,1,0.6], 4:[1,1,0,0.6]}
                    rgba_mask = np.zeros((*pred_slice.shape, 4))
                    for cls, color in colors.items():
                        rgba_mask[pred_slice == cls] = color

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(t1c_norm, cmap='gray', origin='lower')
                    axes[0].set_title("Original")
                    axes[1].imshow(t1c_norm, cmap='gray', origin='lower')
                    axes[1].imshow(rgba_mask, origin='lower')
                    axes[1].set_title("Prediction Overlay")
                    for ax in axes: ax.axis("off")
                    st.pyplot(fig)

                    make_video(CASE_ID)

                    video_path = os.path.abspath(f"./output/video/3dvid_{CASE_ID}_pred_True.mp4")
                    if os.path.exists(video_path):
                        st.subheader("📽️ 종양 3D 영상")
                        st.video(video_path)
                    else:
                        st.warning(f"❗ 영상 파일이 존재하지 않습니다: {video_path}")

# ========================
# 전체 대시보드
# ========================
def show_dashboard():
    show_read_complete_section()   # 첫 번째 섹션 (상단)

    st.markdown("---")  # 제목만 따로 적어줌

def show_main_dashboard():
    df = load_patient_data()
    filtered_df = df[df["doctor_id"].astype(str) == st.session_state.user_id]
    completed_df = filtered_df[filtered_df["read_status"] == True]

    with st.sidebar:
        st.markdown("**의료진 메뉴**")
        st.markdown(f"**ID:** `{st.session_state.user_id}`")

        # ✅ 날짜 포맷 통일
        completed_df["scan_date"] = pd.to_datetime(completed_df["scan_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        completed_df["reading_date"] = pd.to_datetime(completed_df["reading_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d")
        completed_df["followup_date"] = pd.to_datetime(completed_df["followup_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d")

        st.subheader(f"🧠 판독 완료: {len(completed_df)}건")
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder
            gb = GridOptionsBuilder.from_dataframe(completed_df[["patient_id", "name", "age", "scan_date", "reading_date", "followup_date"]])
            gb.configure_selection("single", use_checkbox=True)
            grid_response = AgGrid(completed_df[["patient_id", "name", "age", "scan_date", "reading_date", "followup_date"]],
                                   gridOptions=gb.build(), height=400, update_mode="SELECTION_CHANGED", theme="streamlit")
            selected_rows = grid_response.get("selected_rows", [])
            if selected_rows:
                selected_row = selected_rows[0]
                if "patient_id" in selected_row:
                    st.session_state.selected_patient_id = selected_row["patient_id"]
                    st.rerun()
        except:
            st.dataframe(completed_df)

        if st.button("🔓 로그아웃"):
            st.session_state.logout = True
            st.session_state.logged_in = False
            st.rerun()

    show_dashboard()

def show_login_page():
    st.title("🔐 로그인")
    id = st.text_input("의료진번호")
    password = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        if verify_user(id, password):
            st.session_state.logged_in = True
            st.session_state.user_id = id
            st.rerun()
        else:
            st.error("❌ 로그인 실패")

def show_app():
    if not st.session_state.get("logged_in", False):
        show_login_page()
    else:
        show_main_dashboard()


# #✅ Step 1: 메모 정리 함수 정의
# import re
#
# def clean_memo_file(patient_id):
#     """메모 파일에서 종양 정보(📏) 제거"""
#     path = get_memo_file_path(patient_id)
#     if os.path.exists(path):
#         with open(path, 'r', encoding='utf-8') as f:
#             memo_data = json.load(f)
#         content = memo_data.get('content', '')
#         cleaned = re.sub(r'^📏.*?\n\n?', '', content, flags=re.MULTILINE)
#         memo_data['content'] = cleaned.strip()
#         with open(path, 'w', encoding='utf-8') as f:
#             json.dump(memo_data, f, ensure_ascii=False, indent=4)
#         print(f"✅ 정리 완료: {patient_id}")
#
# # ✅ Step 1: 정리 실행 (환자 전체 대상)
# df = load_patient_data()
# for row in df.itertuples():
#     clean_memo_file(row.patient_id)


show_app()