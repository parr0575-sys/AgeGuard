# app.py
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

# -----------------------------
# 1️⃣ YOLO 모델 로드
# -----------------------------
model = YOLO("yolov8n.pt")  # pretrained COCO 모델

# COCO 클래스 이름
class_names = {
    0: "사람",
    44: "칼"   # knife 클래스가 모델에 포함되어 있으면
}

# -----------------------------
# 2️⃣ 사용자 영상 업로드
# -----------------------------
uploaded_file = st.file_uploader("영상 선택 (MP4)", type=["mp4"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # 임시 출력 파일
    processed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = processed_file.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H264 코덱 환경에 따라 변경 가능
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    frame_skip = 2  # 속도 최적화

    st.text("영상 처리 중... 잠시만 기다려주세요.")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    while ret:
        if frame_count % frame_skip == 0:
            results = model(frame)
            for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                cls = int(cls)
                if cls in class_names and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    label = class_names[cls]
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    # 바운딩 박스
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # 라벨 표시
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # 블러 처리
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    frame[y1:y2, x1:x2] = blurred_roi

        out.write(frame)
        frame_count += 1
        ret, frame = cap.read()

        if frame_count % 10 == 0:
            progress_text.text(f"처리 프레임: {frame_count}")
            progress_bar.progress(min(frame_count / 100, 1.0))

    cap.release()
    out.release()

    st.success("✅ 영상 처리 완료!")
    
    # -----------------------------
    # 3️⃣ 브라우저에서 재생
    # -----------------------------
    st.video(output_path)
