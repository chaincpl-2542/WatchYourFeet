import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Use the appropriate segmentation model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)

    # Create a black background image with the same size as the frame
    bg_removed = np.zeros_like(frame)

    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Expected shape: [N, H_mask, W_mask]
            boxes = result.boxes

            for idx, mask in enumerate(masks):
                # Convert mask to binary mask and type uint8
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Resize mask if its size doesn't match the frame
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Check class id; assuming class 0 is person
                cls_id = int(boxes[idx].cls[0])
                if cls_id == 0:
                    # Use the mask to extract the person from the frame
                    person = cv2.bitwise_and(frame, frame, mask=mask)
                    bg_removed = cv2.add(bg_removed, person)

    cv2.imshow("Background Removed", bg_removed)
    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
