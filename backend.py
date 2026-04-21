import cv2
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import numpy as np

# -------------------------
# Setup
# -------------------------
app = Flask(__name__)
CORS(app)

model = YOLO("yolov8s.pt")
reader = easyocr.Reader(['en'], gpu=True)

cap = cv2.VideoCapture("http://172.16.3.149:8080/video")

# -------------------------
# Config
# -------------------------
CONF_THRESHOLD = 0.5

latest_result = {
    "timestamp": None,
    "motion": False,
    "objects": [],
    "text": []
}

run_inference = True

# -------------------------
# OCR targets
# -------------------------
OCR_OBJECTS = {
    "book", "laptop", "cell phone", "tv",
    "monitor", "screen", "sign", "keyboard", "paper"
}

def should_run_ocr(labels):
    return any(obj in OCR_OBJECTS for obj in labels)

# -------------------------
# Motion detection (fast filter)
# -------------------------
def detect_motion(frame1, frame2, threshold=25):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)

    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh)

    return motion_score > 50000  # tune per camera

# -------------------------
# OCR preprocessing
# -------------------------
def preprocess_for_ocr(img):
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

def upscale(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# -------------------------
# Vision loop
# -------------------------
def vision_loop():
    global latest_result

    ret, prev_frame = cap.read()
    if not ret:
        print("Camera not accessible")
        return

    while True:
        if not run_inference:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        # -------------------------
        # STEP 0: MOTION CHECK
        # -------------------------
        motion = detect_motion(prev_frame, frame)

        if not motion:
            latest_result = {
                "timestamp": time.time(),
                "motion": False,
                "objects": [],
                "text": []
            }
            prev_frame = frame
            time.sleep(0.1)
            continue

        # -------------------------
        # STEP 1: YOLO (ONLY IF MOTION)
        # -------------------------
        results = model(frame, verbose=False)[0]

        labels = []
        ocr_texts = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # CONFIDENCE FILTER
            if conf < CONF_THRESHOLD:
                continue

            label = model.names[cls_id]

            labels.append({
                "label": label,
                "confidence": round(conf, 2)
            })

            # -------------------------
            # STEP 2: OCR IF NEEDED
            # -------------------------
            if label not in OCR_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = upscale(crop)
            crop = preprocess_for_ocr(crop)

            if crop is None:
                continue

            text = reader.readtext(crop, detail=0)

            if text:
                ocr_texts.extend(text)

        latest_result = {
            "timestamp": time.time(),
            "motion": True,
            "objects": labels,
            "text": list(set(ocr_texts))
        }

        prev_frame = frame
        time.sleep(0.15)

# -------------------------
# API ROUTES
# -------------------------
@app.route("/latest")
def latest():
    return jsonify({
        "success": True,
        "data": latest_result
    })

@app.route("/toggle", methods=["POST"])
def toggle():
    global run_inference
    run_inference = not run_inference

    return jsonify({
        "running": run_inference
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
    })

# -------------------------
# START SERVER
# -------------------------
if __name__ == "__main__":
    threading.Thread(target=vision_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)