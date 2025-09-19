import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

# -----------------------------
# 1️⃣ YOLO 모델 로드
# -----------------------------
model = YOLO("yolov8n.pt")  # pretrained COCO 모델

# COCO 클래스 이름 (간단히 사람/칼만)
class_names = {0: "사람", 44: "칼"}

# -----------------------------
# 2️⃣ 사용자가 업로드한 영상 받기
# -----------------------------
uploaded_file = st.file_uploader("MP4 영상을 업로드하세요", type=["mp4"])

if uploaded_file is not None:
    # 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # 결과 파일
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = out_file.name

    # -----------------------------
    # 3️⃣ 영상 열기 & VideoWriter 준비
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("영상 파일을 열 수 없습니다.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    # H.264 코덱으로 브라우저 호환성 향상
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # -----------------------------
    # 4️⃣ 프레임 처리
    # -----------------------------
    frame_count = 0
    frame_skip = 2  # 속도 최적화

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while ret:
        if frame_count % frame_skip == 0:
            results = model(frame)
            for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                cls = int(cls)
                if cls in class_names and conf > 0.5:
                    label = class_names[cls]
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if cls==0 else (0,0,255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # 바운딩 박스 안 블러
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (51,51), 0)
                    frame[y1:y2, x1:x2] = blurred_roi
                    # 라벨 텍스트
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        frame_count += 1
        ret, frame = cap.read()

        # 진행 바 업데이트
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success(f"✅ 영상 처리 완료! 프레임 수: {frame_count}")

    # -----------------------------
    # 5️⃣ 브라우저에서 재생
    # -----------------------------
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    # -----------------------------
    # 6️⃣ 다운로드 버튼
    # -----------------------------
    st.download_button("⬇️ 영상 다운로드", data=video_bytes, file_name="result.mp4", mime="video/mp4")
