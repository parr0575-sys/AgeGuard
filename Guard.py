# Guard_web.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from moviepy.editor import VideoFileClip, vfx
import numpy as np

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

# -------------------------
# 1️⃣ YOLO 모델 로드
# -------------------------
model = YOLO("yolov8n.pt")  # COCO pretrained 모델

# COCO 클래스 이름
class_names = {0: "사람", 44: "칼"}

# -------------------------
# 2️⃣ 사용자 영상 업로드
# -------------------------
uploaded_file = st.file_uploader("MP4 파일 업로드", type=["mp4"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # -------------------------
    # 3️⃣ Video 처리
    # -------------------------
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    processed_frames = []

    frame_count = 0
    frame_skip = 2  # 속도 최적화

    while ret:
        if frame_count % frame_skip == 0:
            results = model(frame)
            for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                cls = int(cls)
                if cls in class_names and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    label = class_names[cls]
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    # 바운딩 박스 + 라벨
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    # 블러링
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    frame[y1:y2, x1:x2] = blurred_roi

        processed_frames.append(frame)
        frame_count += 1
        ret, frame = cap.read()

    cap.release()

    # -------------------------
    # 4️⃣ MoviePy로 영상 저장
    # -------------------------
    st.info("처리 중... 잠시만 기다려주세요.")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    def make_frame(t):
        idx = min(int(t * fps), len(processed_frames) - 1)
        return cv2.cvtColor(processed_frames[idx], cv2.COLOR_BGR2RGB)

    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(lambda frame, t=0: make_frame(t))
    processed_clip.write_videofile(output_file, fps=fps, codec="libx264", audio=False)

    # -------------------------
    # 5️⃣ Streamlit에서 재생
    # -------------------------
    st.success("✅ 영상 처리 완료!")
    st.video(output_file)
