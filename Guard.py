# Guard.py
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

# -----------------------------
# 1️⃣ 모델 로드
# -----------------------------
model = YOLO("yolov8n.pt")  # pretrained COCO
class_names = {0: "person", 44: "knife"}  # 영어 자연어 라벨

# -----------------------------
# 2️⃣ 사용자 영상 업로드
# -----------------------------
uploaded_file = st.file_uploader("MP4 영상을 선택하세요", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

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

        while ret:
            if frame_count % frame_skip == 0:
                results = model(frame)
                for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    cls = int(cls)
                    if cls in class_names and conf > 0.5:
                        label = class_names[cls]
                        x1, y1, x2, y2 = map(int, box)

                        # 1️⃣ 바운딩 박스
                        color = (0,255,0) if cls==0 else (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                        # 2️⃣ 자연어 레이블 (영어로)
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # 3️⃣ 블러 처리
                        roi = frame[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (51,51), 0)
                        frame[y1:y2, x1:x2] = blurred_roi

            out.write(frame)
            frame_count += 1
            ret, frame = cap.read()

        cap.release()
        out.release()
        st.info(f"영상 처리 완료! 총 {frame_count} 프레임 처리됨.")

        # -----------------------------
        # 4️⃣ MoviePy 재인코딩 & 브라우저 재생
        # -----------------------------
        clip = VideoFileClip(output_path)
        final_path = output_path.replace(".mp4","_final.mp4")
        clip.write_videofile(final_path, codec="libx264", audio_codec="aac")

        st.video(final_path)
