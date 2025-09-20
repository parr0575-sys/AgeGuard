# Guard.py
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# -----------------------------
# 🌟 Streamlit UI 스타일
# -----------------------------
st.set_page_config(page_title="Violence Detection", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1f2937, #4b5563);
        color: #f9fafb;
    }
    h1 {
        text-align: center;
        color: #f59e0b;
        font-size: 3em;
        font-family: 'Segoe UI', sans-serif;
    }
    h4 {
        text-align: center;
        color: #fbbf24;
    }
    .stFileUploader label {
        font-weight: bold;
        color: #f9fafb;
    }
    .stButton>button {
        background-color: #f59e0b;
        color: #111827;
        font-weight: bold;
        border-radius: 10px;
        padding: 8px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 1️⃣ 타이틀 영역
# -----------------------------
st.title("🛡️ 영상 폭력 탐지 & 블러링 WebApp")
st.markdown("#### 업로드한 영상에서 **폭력 장면(칼, 총)** 을 탐지하고 자동 블러링합니다.")
st.divider()

# -----------------------------
# 2️⃣ YOLO 모델 로드
# -----------------------------
# 학습 완료된 칼/총 탐지 모델 (.pt) 사용
# 학습 파일이 없으면 임시로 yolov8n.pt 사용
import os
model_path = "weapon_knife_model.pt"
if not os.path.exists(model_path):
    st.warning("⚠️ weapon_knife_model.pt 파일이 없습니다. 임시 COCO pretrained 모델 사용")
    model_path = "yolov8n.pt"

model = YOLO(model_path)
class_names = {44: "knife", 45: "gun"}  # 필요시 클래스 번호에 맞춰 수정

# -----------------------------
# 3️⃣ 사용자 영상 업로드
# -----------------------------
st.markdown("""
<div style='background-color:#111827; padding:15px; border-radius:15px; text-align:center; margin-bottom:10px;'>
    <p style='color:#f9fafb; font-size:18px;'>📂 MP4 영상을 선택하세요</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    st.info("⏳ 영상 처리 중... 잠시만 기다려주세요.")

    # -----------------------------
    # 4️⃣ 영상 처리
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("🚨 영상 파일을 열 수 없습니다.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        frame_skip = 2  # 속도 최적화
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0, text="프레임 처리 중...")

        while ret:
            if frame_count % frame_skip == 0:
                results = model(frame)
                for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    cls = int(cls)
                    if cls in class_names and conf > 0.5:
                        label = class_names[cls]
                        x1, y1, x2, y2 = map(int, box)

                        # 바운딩 박스 + 라벨
                        color = (255,0,0)  # 빨간색 강조
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # 블러 처리
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            blurred_roi = cv2.GaussianBlur(roi, (51,51), 0)
                            frame[y1:y2, x1:x2] = blurred_roi

            out.write(frame)
            frame_count += 1
            ret, frame = cap.read()

            # 진행률 업데이트
            progress.progress(min(frame_count / total_frames, 1.0), text=f"{frame_count}/{total_frames} 프레임 처리")

        cap.release()
        out.release()
        st.success(f"✅ 영상 처리 완료! 총 {frame_count} 프레임 분석됨.")

        # -----------------------------
        # 5️⃣ 결과 영상 표시 & 다운로드
        # -----------------------------
        clip = VideoFileClip(output_path)
        final_path = output_path.replace(".mp4","_final.mp4")
        clip.write_videofile(final_path, codec="libx264", audio_codec="aac")

        st.video(final_path)
        st.download_button(
            label="📥 결과 영상 다운로드",
            data=open(final_path, "rb").read(),
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
