import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# โหลดโมเดล YOLOv8 (ใช้ model ขนาดเล็กเพื่อความรวดเร็ว)
model = YOLO('yolov8n.pt')

# สร้างตัวติดตาม SORT
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# เปิดกล้อง (หรือใช้วิดีโอไฟล์ก็ได้)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้ YOLO ตรวจจับวัตถุในเฟรม
    results = model(frame)
    detections = []

    # วนลูปอ่านผลการตรวจจับ (ผลลัพธ์อาจมีหลาย object)
    for result in results:
        # ดึง bounding boxes, confidence และ class id จากผลลัพธ์
        boxes = result.boxes.xyxy.cpu().numpy()  # รูปแบบ [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()   # ค่า confidence
        classes = result.boxes.cls.cpu().numpy()   # class id

        # กรองเฉพาะคน (ในโมเดล COCO, class id ของ 'person' คือ 0)
        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) == 0 and score > 0.5:
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, score])

    # เปลี่ยน detections ให้เป็น numpy array (ถ้าไม่ว่างเปล่า)
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # อัปเดตตัวติดตามด้วยการตรวจจับใหม่ในแต่ละเฟรม
    tracks = tracker.update(detections)

    # วาด bounding boxes และ ID ที่ติดตามได้บนเฟรม
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("YOLO with SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
