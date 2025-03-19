import cv2
import numpy as np

def order_points(pts):
    # จัดเรียงจุดให้อยู่ในลำดับ: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

cap = cv2.VideoCapture(0)

# กำหนด HSV range จากสีที่ต้องการ (ปรับแต่งได้ตามสภาพแสง)
lower_color = np.array([75, 40, 200])
upper_color = np.array([85, 100, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # สร้าง mask สำหรับสีที่ต้องการ
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # ทำ Morphological Operation เพื่อลด noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # ค้นหา Contours ใน mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # เลือก contour ที่มีพื้นที่มากที่สุด
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # ถ้า contour มี 4 จุด (สี่เหลี่ยม) ถือว่าเป็นพื้นที่ต้องการ
        if len(approx) == 4:
            # วาด contour ที่ตรวจจับได้ลงบนภาพต้นฉบับ
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            
            # จัดเรียงจุดให้อยู่ในลำดับที่ถูกต้อง
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            # กำหนดจำนวน grid ที่ต้องการ (ตัวอย่าง 3x3)
            gridRows, gridCols = 3, 3

            # วาดเส้น grid แนวนอน
            for i in range(1, gridRows):
                # คำนวณจุดเริ่มต้นและจุดสิ้นสุดของเส้นโดยใช้การอินเทอร์โพเลชันระหว่างมุมซ้ายและขวาของแต่ละแถว
                alpha = i / gridRows
                start = tl + (bl - tl) * alpha
                end   = tr + (br - tr) * alpha
                start = tuple(start.astype(int))
                end   = tuple(end.astype(int))
                cv2.line(frame, start, end, (255, 0, 0), 2)

            # วาดเส้น grid แนวตั้ง
            for j in range(1, gridCols):
                beta = j / gridCols
                start = tl + (tr - tl) * beta
                end   = bl + (br - bl) * beta
                start = tuple(start.astype(int))
                end   = tuple(end.astype(int))
                cv2.line(frame, start, end, (255, 0, 0), 2)
    
    # แสดงผลภาพ
    cv2.imshow("Color Mask", mask)
    cv2.imshow("Detected Floor with Grid", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
