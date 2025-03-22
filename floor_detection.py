import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

cap = cv2.VideoCapture(0)

lower_color = np.array([75, 40, 200])
upper_color = np.array([85, 100, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            gridRows, gridCols = 3, 3

            for i in range(1, gridRows):
                alpha = i / gridRows
                start = tl + (bl - tl) * alpha
                end   = tr + (br - tr) * alpha
                start = tuple(start.astype(int))
                end   = tuple(end.astype(int))
                cv2.line(frame, start, end, (255, 0, 0), 2)

            for j in range(1, gridCols):
                beta = j / gridCols
                start = tl + (tr - tl) * beta
                end   = bl + (br - bl) * beta
                start = tuple(start.astype(int))
                end   = tuple(end.astype(int))
                cv2.line(frame, start, end, (255, 0, 0), 2)
    
    cv2.imshow("Color Mask", mask)
    cv2.imshow("Detected Floor with Grid", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
