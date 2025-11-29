import cv2
import numpy as np
import math
import time
import csv

MAIN_WIN = "Advanced Multi-Color Distance Detector"
FOCAL_LENGTH = None      # Set this if you later calibrate
KNOWN_DISTANCE_CM = 30.0
CSV_FILE = "distance_log.csv"

# Predefined color ranges (HSV)
color_ranges = {
    "Red":    [(0, 120, 70), (10, 255, 255)],
    "Blue":   [(94, 80, 2),  (126, 255, 255)],
    "Green":  [(35, 100, 100), (85, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Black":  [(0, 0, 0), (179, 255, 30)]
}

# Drawing colors (BGR)
draw_colors = {
    "Red":    (0, 0, 255),
    "Blue":   (255, 0, 0),
    "Green":  (0, 255, 0),
    "Yellow": (0, 255, 255),
    "Black":  (0, 0, 0)
}

selected_color = None
mouse_params = {"frame": None}
cap = None
fps_start = time.time()
frame_count = 0


def get_centroid(mask):
    """Find centroid of largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:  # minimum area filter
            return None
        M = cv2.moments(c)
        if M["m00"] != 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None


def on_mouse(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel = hsv[y, x]
        h, s, v = pixel

        lower = (max(h - 10, 0), max(s - 50, 0), max(v - 50, 0))
        upper = (min(h + 10, 179), min(s + 50, 255), min(v + 50, 255))

        color_name = f"Color{len(color_ranges) + 1}"
        color_ranges[color_name] = [lower, upper]
        draw_colors[color_name] = (
            int(frame[y, x][0]),
            int(frame[y, x][1]),
            int(frame[y, x][2])
        )
        selected_color = color_name
        print(f"Added new color: {color_name} - HSV Range: {lower}-{upper}")


def calculate_focal_length(pixel_distance, known_distance_cm, real_width_cm):
    return (pixel_distance * known_distance_cm) / real_width_cm


def calculate_distance(pixel_distance):
    if FOCAL_LENGTH is None:
        # Fallback: just scale pixels so you still see changing numbers
        return pixel_distance / 50.0
    real_width_cm = 5.0  # Adjust if using a real known width
    return (real_width_cm * FOCAL_LENGTH) / pixel_distance


def init_csv():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Object1", "Object2", "Distance_cm"])


# ---- Camera selection ----
print("Select Camera:")
print("1. Laptop Camera")
print("2. Phone Camera (IP)")
choice = input("Enter choice: ")

if choice == "2":
    ip = input("Enter IP camera URL (e.g., http://192.168.x.x:8080/video): ")
    cap = cv2.VideoCapture(ip)
else:
    cap = cv2.VideoCapture(0)

cv2.namedWindow(MAIN_WIN)
cv2.setMouseCallback(MAIN_WIN, on_mouse, param=mouse_params)
init_csv()

# ---- Main loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mouse_params["frame"] = frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    centers = {}

    # Color detection + centroid
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        center = get_centroid(mask)
        if center:
            centers[color] = center
            cv2.circle(frame, center, 10, draw_colors[color], -1)
            cv2.putText(frame, color, (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_colors[color], 2)

    # Distance between all detected centers
    keys = list(centers.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            pt1, pt2 = centers[k1], centers[k2]
            pixel_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            dist_cm = calculate_distance(pixel_dist)
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"{dist_cm:.1f}cm", mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Log to CSV
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%H:%M:%S"), k1, k2, f"{dist_cm:.2f}"])

    # FPS calculation
    frame_count += 1
    if time.time() - fps_start >= 1:
        fps = frame_count / (time.time() - fps_start)
        frame_count = 0
        fps_start = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(MAIN_WIN, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
