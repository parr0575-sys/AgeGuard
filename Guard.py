import cv2
from ultralytics import YOLO
import os

# -----------------------------
# 1️⃣ YOLO 모델 로드
# -----------------------------
model = YOLO("yolov8n.pt")  # pretrained YOLO 모델

# COCO 클래스 이름 (영어)
class_names = {
    0: "Person",
    44: "Knife"
}

# -----------------------------
# 2️⃣ 입력 / 출력 영상 경로
# -----------------------------
video_path = r"C:\Users\parr0\OneDrive\바탕 화면\174.mp4"  # 본인 영상 경로
output_path = r"C:\Users\parr0\OneDrive\바탕 화면\result.mp4"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# -----------------------------
# 3️⃣ 영상 열기 & VideoWriter 준비
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# -----------------------------
# 4️⃣ 프레임 처리
# -----------------------------
frame_count = 0
frame_skip = 2  # 2프레임마다 처리 -> 속도 최적화

while ret:
    if frame_count % frame_skip == 0:
        results = model(frame)
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            cls = int(cls)
            if cls in class_names and conf > 0.5:
                label = class_names[cls]
                x1, y1, x2, y2 = map(int, box)
                
                # 바운딩 박스
                color = (0, 255, 0) if cls==0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 레이블 출력
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)
                
                # 객체 영역 블러 처리
                roi = frame[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (51,51), 0)
                frame[y1:y2, x1:x2] = blurred_roi

    out.write(frame)
    frame_count += 1
    ret, frame = cap.read()

    if frame_count % 50 == 0:
        print(f"처리 프레임: {frame_count}")

cap.release()
out.release()
print(f"✅ 완료: {frame_count} 프레임 처리, 결과 영상 → {output_path}")
