"""
Road Surface Defect Monitoring — Flask Backend
Supports car model + drone model switching, inference, and live metrics.
Run: python app.py
"""

import os, json, io, base64, time
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Model registry ──────────────────────────────────────────────
MODELS = {
    "car":   {"path": None, "model": None, "label": "Car-mounted",  "icon": "ti-car"},
    "drone": {"path": None, "model": None, "label": "Drone aerial", "icon": "ti-drone"},
}

CLASSES = ["Pothole_Severe", "Smooth", "Crack_Mild", "Pothole_Mild", "Crack_Severe"]
SEVERITY = {"Pothole_Severe": 10, "Crack_Severe": 7, "Pothole_Mild": 5,
            "Crack_Mild": 3, "Smooth": 0}
COLORS   = {"Pothole_Severe": "#e24b4a", "Crack_Severe": "#ba7517",
            "Pothole_Mild": "#ef9f27", "Crack_Mild": "#639922", "Smooth": "#1d9e75"}

# Rolling in-memory log (last 200 records)
_log: deque = deque(maxlen=200)

# ── Helpers ─────────────────────────────────────────────────────

def priority(score: float) -> str:
    if score >= 8.0: return "CRITICAL"
    if score >= 4.0: return "HIGH"
    if score >= 1.5: return "MEDIUM"
    return "LOW"

def load_model(key: str, path: str):
    from ultralytics import YOLO
    MODELS[key]["path"]  = path
    MODELS[key]["model"] = YOLO(path)
    print(f"[Model] Loaded {key} from {path}")

def run_inference(key: str, img_bytes: bytes, conf: float = 0.25) -> dict:
    from ultralytics import YOLO
    import tempfile, cv2

    m = MODELS[key]["model"]
    if m is None:
        return {"error": f"{key} model not loaded"}

    # Write bytes to temp file (YOLO needs path or numpy)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    results = m.predict(frame, conf=conf, imgsz=640, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class":      r.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox":       [round(x, 1) for x in box.xyxy[0].tolist()],
                "color":      COLORS.get(r.names[cls_id], "#888"),
            })

    # Draw boxes on frame and encode to base64
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = det["color"].lstrip("#")
        bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    score = sum(SEVERITY.get(d["class"], 0) * d["confidence"] for d in detections)
    score = round(score, 3)

    record = {
        "id":             len(_log) + 1,
        "timestamp":      datetime.utcnow().isoformat(),
        "model_key":      key,
        "model_label":    MODELS[key]["label"],
        "detections":     detections,
        "priority_score": score,
        "priority_level": priority(score),
    }
    _log.append(record)
    return {**record, "annotated_image": img_b64}


# ── Routes ───────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify({
        "models": {
            k: {
                "loaded": v["model"] is not None,
                "path":   v["path"],
                "label":  v["label"],
            }
            for k, v in MODELS.items()
        },
        "log_count": len(_log),
    })

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.json or {}
    key  = data.get("key")
    path = data.get("path", "").strip()
    if key not in MODELS:
        return jsonify({"error": "Invalid model key"}), 400
    if not path or not Path(path).exists():
        return jsonify({"error": f"File not found: {path}"}), 400
    try:
        load_model(key, path)
        return jsonify({"ok": True, "message": f"{MODELS[key]['label']} model loaded."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/infer", methods=["POST"])
def api_infer():
    key  = request.form.get("model", "car")
    conf = float(request.form.get("conf", 0.25))
    if key not in MODELS:
        return jsonify({"error": "bad model key"}), 400
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400
    img_bytes = request.files["image"].read()
    result = run_inference(key, img_bytes, conf)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

@app.route("/api/log")
def api_log():
    n = int(request.args.get("n", 50))
    return jsonify(list(_log)[-n:])

@app.route("/api/stats")
def api_stats():
    records = list(_log)
    if not records:
        return jsonify({"total": 0, "by_class": {}, "by_priority": {}, "by_model": {}})

    by_class    = {}
    by_priority = {}
    by_model    = {}

    for rec in records:
        p = rec["priority_level"]
        m = rec["model_key"]
        by_priority[p] = by_priority.get(p, 0) + 1
        by_model[m]    = by_model.get(m, 0) + 1
        for det in rec.get("detections", []):
            c = det["class"]
            by_class[c] = by_class.get(c, 0) + 1

    return jsonify({
        "total":       len(records),
        "by_class":    by_class,
        "by_priority": by_priority,
        "by_model":    by_model,
        "avg_score":   round(sum(r["priority_score"] for r in records) / len(records), 3),
        "critical_pct": round(by_priority.get("CRITICAL", 0) / len(records) * 100, 1),
    })

@app.route("/api/clear_log", methods=["POST"])
def api_clear_log():
    _log.clear()
    return jsonify({"ok": True})

@app.route("/")
def index():
    # Change "static" to "."
    return send_from_directory(".", "index.html")

# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Road Defect Monitor — Backend")
    print("  http://localhost:5000")
    print("  Load models via /api/load_model or the dashboard UI")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
