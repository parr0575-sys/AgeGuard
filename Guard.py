import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.title("Video Violence Detection & Blurring")

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
if uploaded_file:

    # 1️⃣ 업로드 영상 임시 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)  # 업로드 영상 재생

    st.info("Processing video... This may take a while depending on video length.")

    # 2️⃣ YOLO 모델 로드
    model = YOLO("yolov8n.pt")
    # 영문/한글 레이블 선택 가능
    class_names = {0: "사람", 44: "칼"}  # 한글 사용
    # class_names = {0: "Person", 44: "Knife"}  # 영문 사용 시

    # 3️⃣ 결과 영상 임시 파일
    output_path = os.path.join(tempfile.gettempdir(), "result.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video!")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        frame_skip = 2  # 속도 최적화

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # 폰트 설정 (한글 표시용)
        font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기본 한글 폰트
        font_size = 20

        # ---------------- 프레임 처리 ----------------
        while ret:
            if frame_count % frame_skip == 0:
                results = model(frame)
                for box, cls, conf in zip(results[0].boxes.xyxy,
                                          results[0].boxes.cls,
                                          results[0].boxes.conf):
                    cls = int(cls)
                    if cls in class_names and conf > 0.5:
                        label = class_names[cls]
                        x1, y1, x2, y2 = map(int, box)

                        color = (0,255,0) if cls==0 else (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                        # ----------------- 한글/영문 레이블 표시 -----------------
                        # OpenCV 대신 PIL 사용
                        img_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                        except:
                            font = ImageFont.load_default()
                        draw.text((x1, y1-25), label, font=font, fill=color)
                        frame = np.array(img_pil)

                        # 블러 처리
                        roi = frame[y1:y2, x1:x2]
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51,51), 0)

            out.write(frame)
            frame_count += 1
            ret, frame = cap.read()

            # 진행률 업데이트
            progress_text.text(f"Processing frame {frame_count}/{total_frames}")
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # ---------------- 처리 완료 후 ----------------
        cap.release()
        out.release()

        st.success("✅ Video processing completed!")
        st.video(output_path)  # 처리된 영상 재생

        # 다운로드 버튼
        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="result.mp4")
