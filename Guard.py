import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import io

st.title("영상 폭력/비폭력 탐지 & 블러링 WebApp")

model = YOLO("yolov8n.pt")
class_names = {0: "사람", 44: "칼"}

uploaded_file = st.file_uploader("MP4 영상 업로드", type=["mp4"])

if uploaded_file is not None:
    # 업로드된 파일 임시 저장
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # 결과 영상 임시 저장
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("영상 파일을 열 수 없습니다.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        while ret:
            results = model(frame)
            for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                cls = int(cls)
                if cls in class_names and conf > 0.5:
                    label = class_names[cls]
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    frame[y1:y2, x1:x2] = blurred_roi
            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        st.success("영상 처리 완료!")

        # ------------------------
        # 6️⃣ 웹에서 바로 재생
        # ------------------------
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
