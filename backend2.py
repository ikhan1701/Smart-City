import cv2
import threading
import time
import base64
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app)

# Models
yolo = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vlm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
vlm.to(device)

cap = cv2.VideoCapture("http://172.16.3.198:8080/video")

latest_result = {
    "timestamp": None,
    "objects": [],
    "image": None
}

CONF_THRESHOLD = 0.4

# -------------------------
# Encode image
# -------------------------
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------
# VLM caption (REAL)
# -------------------------
def generate_caption(crop):
    try:
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except:
        return "No description available"

# -------------------------
# Vision loop
# -------------------------
def vision_loop():
    global latest_result

    last_vlm_time = 0
    VLM_INTERVAL = 3  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = yolo(frame, verbose=False)[0]

        objects = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            # 🔥 Only run VLM occasionally
            description = ""
            if time.time() - last_vlm_time > VLM_INTERVAL and crop.size > 0:
                description = generate_caption(crop)
                last_vlm_time = time.time()

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2],
                "description": description
            })

        latest_result = {
            "timestamp": time.time(),
            "objects": objects,
            "image": encode_frame(frame)
        }

        time.sleep(0.1)

# -------------------------
# API
# -------------------------
@app.route("/latest")
def latest():
    return jsonify(latest_result)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    threading.Thread(target=vision_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)