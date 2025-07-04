# 📹 CCTV Monitoring System with Pose Estimation and Restricted Zone Detection

This project is a real-time CCTV monitoring system using **YOLOv8 Pose Estimation** that detects human poses, tracks individuals, and identifies violations when a person enters a restricted zone in a factory or workspace.

---

## ✅ Features

* 🔍 **Person Detection** using YOLOv8 Pose model
* 🦵 **Pose Estimation** for tracking specific keypoints (e.g., ankles)
* 🚧 **Restricted Area Detection** via polygonal zones
* 📊 **Violation Logging** with frame number, zone name, and location (CSV output)
* 🖼️ **Real-time Display** with bounding boxes and alert overlays

---

## 🧾 Requirements

Make sure you have Python 3.10 or later. Install dependencies:

```bash
pip install ultralytics opencv-python pandas numpy
```

> **Note:** If you face DLL issues related to OpenMP (e.g., `libiomp5md.dll`), add this to the top of `app.py`:
>
> ```python
> import os
> os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
> ```

---

## 📂 Project Structure

```bash
📁 cctv-monitoring-system
├── app.py                  # Main application
├── zones.json             # JSON file with restricted zone polygons
├── cctv_footage.mp4       # Input CCTV video footage
├── violations.csv         # Output CSV for logged violations
└── README.md              # This file
```

---

## 🎯 How It Works

1. **Load YOLOv8 Pose model** using Ultralytics.
2. **Read video frame-by-frame** using OpenCV.
3. For each frame:

   * Run pose estimation
   * Identify the right ankle keypoint (index 15)
   * Check if ankle is inside any restricted polygon
   * If yes: draw alert, save to CSV
4. Draw all restricted zones with polygon boundaries.
5. Save all violations to `violations.csv`.

---

## 🗂️ zones.json Format

```json
{
  "zones": [
    {
      "name": "Right Zone",
      "polygon": [[500, 300], [800, 300], [800, 700], [500, 700]]
    }
  ]
}
```

You can define multiple zones by adding more objects to the `zones` list.

---

## 🧪 Example Output (violations.csv)

```csv
frame,zone,x,y
42,Right Zone,745,560
78,Right Zone,820,510
```

---

## ▶️ Running the App

```bash
python app.py
```

Press `q` to stop video playback.

---

## 📌 Notes

* If pose detection fails due to lighting, try adjusting `alpha` and `beta` in:

  ```python
  frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
  ```
* You can switch to `yolov8n-pose.pt` for faster but less accurate inference.

---

## 📥 Credits

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [Python](https://python.org)

---

## 🛠️ Future Improvements

* Add person ID tracking
* Send alert notifications (e.g., email/telegram)
* Train custom model for factory-specific behavior
