import cv2
import json
import numpy as np
from ultralytics import YOLO
import pandas as pd
model = YOLO("yolov8m-pose.pt")
violations_log = []

# Load restricted zones

with open("zones.json") as f:
    zone_data = json.load(f)

zones = [(z["name"], np.array(z["polygon"], dtype=np.int32)) for z in zone_data["zones"]]
violations = []

cap = cv2.VideoCapture("cctv_footage.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy()  # shape: [num_people, 17, 2]

    for i, person in enumerate(keypoints):
        ankle = person[15]  # Right ankle (you can use left too)
        x, y = int(ankle[0]), int(ankle[1])

        for name, polygon in zones:
            inside = cv2.pointPolygonTest(polygon, (x, y), False)
            if inside >= 0:
                cv2.putText(frame, f"VIOLATION!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                violations_log.append({
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "zone": name,
                    "x": int(x),
                    "y": int(y)
                })
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

    # Draw zones
    for name, polygon in zones:
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw poses
    annotated = results[0].plot()
    cv2.imshow("CCTV Monitoring", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
df = pd.DataFrame(violations_log)
df.to_csv("violations.csv", index=False)
print("✅ Violations saved to violations.csv")

