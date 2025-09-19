import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

# -----------------------------
# 1️⃣ YOLO 모델 로드
# -----------------------------
model = YOLO("yolov8n.pt")  # pretrained COCO 모델

# COCO 클래스 이름
class_names = {0: "사람", 44: "칼"}

# -----------------------------
# 2️⃣ 영상 업로드
# -----------------------------
uploaded_file = st.file_uploader("MP4 영상 업로드", type=["mp4"])

if uploaded_file is not None:
    # 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # 결과 영상 저장 경로
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # -----------------------------
    # 3️⃣ 영상 열기 & VideoWriter 준비
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("영상 파일을 열 수 없습니다.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        frame_skip = 2  # 속도 최적화

        stframe = st.empty()  # 실시간 웹 표시용

        # -----------------------------
        # 4️⃣ 프레임 처리
        # -----------------------------
        while ret:
            if frame_count % frame_skip == 0:
                results = model(frame)
                for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    cls = int(cls)
                    if cls in class_names and conf > 0.5:
                        label = class_names[cls]
                        x1, y1, x2, y2 = map(int, box)
                        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                        # 바운딩 박스 + 라벨
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        # 블러 처리
                        roi = frame[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        frame[y1:y2, x1:x2] = blurred_roi

            out.write(frame)
            frame_count += 1
            ret, frame = cap.read()

        cap.release()
        out.release()
        st.success(f"영상 처리 완료! 총 {frame_count} 프레임 처리됨.")

        # -----------------------------
        # 5️⃣ 웹에서 영상 재생
        # -----------------------------
        st.video(output_path)
