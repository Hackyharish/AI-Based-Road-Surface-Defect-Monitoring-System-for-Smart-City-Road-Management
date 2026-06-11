"""
Road Surface Defect Monitoring — Flask Backend
Time-Optimized Fleet Routing, Multi-Stage Tracking & Auto-Completion Engine
Run: python app.py
"""

import os, json, io, base64, time, random, math
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Registry & Constants ─────────────────────────────────────────
MODELS = {
    "car":   {"path": None, "model": None, "label": "Car-mounted",  "icon": "ti-car"},
    "drone": {"path": None, "model": None, "label": "Drone aerial", "icon": "ti-drone"},
}

CLASSES = ["Pothole_Severe", "Smooth", "Crack_Mild", "Pothole_Mild", "Crack_Severe"]
SEVERITY = {"Pothole_Severe": 10, "Crack_Severe": 7, "Pothole_Mild": 5, "Crack_Mild": 3, "Smooth": 0}
COLORS   = {"Pothole_Severe": "#e24b4a", "Crack_Severe": "#ba7517", "Pothole_Mild": "#ef9f27", "Crack_Mild": "#639922", "Smooth": "#1d9e75"}
REPAIR_TIMES = {"Pothole_Severe": 3.5, "Crack_Severe": 2.0, "Pothole_Mild": 1.0, "Crack_Mild": 0.5, "Smooth": 0.0}

# Initial Team Locations and Availability Trackers
TEAMS = {
    "Chennai Corp - Zone 8 Unit":   {"lat": 13.0827, "lon": 80.2707, "available_from": datetime.utcnow()},
    "Chennai Corp - Zone 9 Unit":   {"lat": 13.0473, "lon": 80.2479, "available_from": datetime.utcnow()},
    "Rapid Response Squad (Night)": {"lat": 13.0012, "lon": 80.2565, "available_from": datetime.utcnow()},
    "Contractor: BuildTech Infra":  {"lat": 13.1143, "lon": 80.2133, "available_from": datetime.utcnow()},
    "Contractor: PaveRight Ltd":    {"lat": 12.9815, "lon": 80.2180, "available_from": datetime.utcnow()},
    "Metro Highways Division":      {"lat": 13.0604, "lon": 80.2496, "available_from": datetime.utcnow()}
}

_log: deque = deque(maxlen=200)

# ── Helpers ─────────────────────────────────────────────────────

def priority(score: float) -> str:
    if score >= 8.0: return "CRITICAL"
    if score >= 4.0: return "HIGH"
    if score >= 1.5: return "MEDIUM"
    if score >  0.0: return "LOW"
    return "NORMAL"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def check_auto_completion():
    """Lazily evaluates jobs. Transitions to In Progress upon arrival, and Completed upon ETA."""
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    for rec in _log:
        if rec["labor_status"] not in ["Completed", "Unassigned"] and rec.get("eta"):
            # If the final ETA has passed -> Completed
            if now_str >= rec["eta"]:
                rec["labor_status"] = "Completed"
            # If they have arrived on site but haven't finished -> In Progress
            elif rec.get("arrival_time") and now_str >= rec["arrival_time"]:
                if rec["labor_status"] in ["Auto-Dispatched", "Pending Dispatch", "Manual Reassign"]:
                    rec["labor_status"] = "In Progress"

def calculate_auto_eta(team_name, dist_km, detections):
    """Calculates arrival time (travel) and completion time (travel + repair)."""
    travel_hours = dist_km / 20.0 
    repair_hours = max((REPAIR_TIMES.get(d["class"], 0.5) for d in detections), default=0.5) if detections else 0.5
    
    now = datetime.utcnow()
    # Team starts driving either now, or when they finish their previous job
    start_time = max(now, TEAMS[team_name]["available_from"])
    
    arrival_time = start_time + timedelta(hours=travel_hours)
    completion_time = arrival_time + timedelta(hours=repair_hours)
    
    TEAMS[team_name]["available_from"] = completion_time # Lock team until completion
    return arrival_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"), completion_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def load_model(key: str, path: str):
    from ultralytics import YOLO
    MODELS[key]["path"]  = path
    MODELS[key]["model"] = YOLO(path)
    print(f"[Model] Loaded {key} from {path}")

def run_inference_no_dispatch(key: str, img_bytes: bytes, conf: float = 0.25) -> dict:
    from ultralytics import YOLO
    import cv2

    m = MODELS[key]["model"]
    if m is None: return {"error": f"{key} model not loaded"}

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return {"error": "Could not decode image"}

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

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = det["color"].lstrip("#")
        bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    score = sum(SEVERITY.get(d["class"], 0) * d["confidence"] for d in detections)
    score = round(score, 3)

    record = {
        "id":             len(_log) + 1,
        "timestamp":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "model_key":      key,
        "model_label":    MODELS[key]["label"],
        "detections":     detections,
        "priority_score": score,
        "priority_level": priority(score),
        "gps":            {"lat": random.uniform(12.95, 13.15), "lon": random.uniform(80.15, 80.26)},
        "labor_team":     None,
        "labor_status":   "Unassigned", 
        "arrival_time":   None,
        "eta":            None,
        "annotated_image": base64.b64encode(buf).decode()
    }
    _log.append(record)
    return record

def run_inference_on_frame(key: str, frame, conf: float = 0.25) -> dict:
    """Run YOLO inference on a single decoded OpenCV frame (numpy array). Returns detections + annotated frame."""
    import cv2
    m = MODELS[key]["model"]
    if m is None: return {"error": f"{key} model not loaded"}

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

    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = det["color"].lstrip("#")
        bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(annotated, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    return {"detections": detections, "annotated": annotated}


def global_auto_dispatch():
    """Time-Optimized Routing: Assigns jobs to the team that will finish them EARLIEST."""
    unassigned = [r for r in _log if r["labor_status"] == "Unassigned" and r["priority_level"] != "NORMAL"]
    unassigned.sort(key=lambda x: x["priority_score"], reverse=True)
    
    assigned_count = 0
    now = datetime.utcnow()
    
    for rec in unassigned:
        best_team = None
        earliest_completion = None
        
        repair_hours = max((REPAIR_TIMES.get(d["class"], 0.5) for d in rec["detections"]), default=0.5) if rec["detections"] else 0.5

        for team_name, data in TEAMS.items():
            dist = haversine(data["lat"], data["lon"], rec["gps"]["lat"], rec["gps"]["lon"])
            travel_hours = dist / 20.0
            
            start_time = max(now, data["available_from"])
            completion_time = start_time + timedelta(hours=(travel_hours + repair_hours))
            
            if earliest_completion is None or completion_time < earliest_completion:
                earliest_completion = completion_time
                best_team = team_name
                
        if best_team:
            dist = haversine(TEAMS[best_team]["lat"], TEAMS[best_team]["lon"], rec["gps"]["lat"], rec["gps"]["lon"])
            arr_iso, comp_iso = calculate_auto_eta(best_team, dist, rec["detections"])
            
            rec["labor_team"] = best_team
            rec["labor_status"] = "Auto-Dispatched"
            rec["arrival_time"] = arr_iso
            rec["eta"] = comp_iso
            
            # Move the team to this job site
            TEAMS[best_team]["lat"] = rec["gps"]["lat"]
            TEAMS[best_team]["lon"] = rec["gps"]["lon"]
            assigned_count += 1
            
    return assigned_count

# ── Routes ───────────────────────────────────────────────────────

@app.route("/api/infer_video", methods=["POST"])
def api_infer_video():
    """
    Accepts a video file upload. Samples frames at the requested interval,
    runs YOLO on each sampled frame, aggregates detections, auto-dispatches,
    and returns:
      - per-frame summary
      - worst-frame annotated image (base64 JPEG)
      - overall dispatch count
    """
    import cv2, tempfile

    key  = request.form.get("model", "car")
    conf = float(request.form.get("conf", 0.25))
    sample_fps = float(request.form.get("sample_fps", 1.0))  # frames to analyse per second

    if key not in MODELS:
        return jsonify({"error": "Invalid model key"}), 400
    if MODELS[key]["model"] is None:
        return jsonify({"error": f"{MODELS[key]['label']} model not loaded"}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    # Write upload to a temp file so OpenCV can open it
    suffix = Path(video_file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 400

        vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / vid_fps if vid_fps else 0

        frame_interval = max(1, int(vid_fps / sample_fps))  # sample every N frames

        results_per_frame = []
        worst_score  = -1
        worst_img_b64 = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                res = run_inference_on_frame(key, frame, conf)
                if "error" in res:
                    cap.release()
                    return jsonify({"error": res["error"]}), 500

                detections = res["detections"]
                score = sum(SEVERITY.get(d["class"], 0) * d["confidence"] for d in detections)
                score = round(score, 3)
                timestamp_s = round(frame_idx / vid_fps, 2)

                record = {
                    "id":             len(_log) + 1,
                    "timestamp":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "model_key":      key,
                    "model_label":    MODELS[key]["label"],
                    "detections":     detections,
                    "priority_score": score,
                    "priority_level": priority(score),
                    "gps":            {"lat": random.uniform(12.95, 13.15), "lon": random.uniform(80.15, 80.26)},
                    "labor_team":     None,
                    "labor_status":   "Unassigned",
                    "arrival_time":   None,
                    "eta":            None,
                    "video_timestamp_s": timestamp_s,
                }
                _log.append(record)
                results_per_frame.append({k: v for k, v in record.items()})

                if score > worst_score and detections:
                    worst_score = score
                    _, buf = cv2.imencode(".jpg", res["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    worst_img_b64 = base64.b64encode(buf).decode()

            frame_idx += 1

        cap.release()

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    dispatched = global_auto_dispatch()
    check_auto_completion()

    clean = [{k: v for k, v in r.items()} for r in results_per_frame]
    return jsonify({
        "frames_analysed": len(results_per_frame),
        "total_frames":    total_frames,
        "duration_s":      round(duration_s, 2),
        "sample_fps":      sample_fps,
        "inferences":      clean,
        "worst_frame_image": worst_img_b64,
        "dispatched_count": dispatched,
    })


@app.route("/api/status")
def status():
    return jsonify({"models": { k: {"loaded": v["model"] is not None, "path": v["path"], "label": v["label"]} for k, v in MODELS.items() }, "log_count": len(_log), "teams": list(TEAMS.keys())})

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.json or {}
    key, path = data.get("key"), data.get("path", "").strip()
    if key not in MODELS: return jsonify({"error": "Invalid model key"}), 400
    if not path or not Path(path).exists(): return jsonify({"error": f"File not found: {path}"}), 400
    try:
        load_model(key, path)
        return jsonify({"ok": True, "message": f"{MODELS[key]['label']} model loaded."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/infer", methods=["POST"])
def api_infer():
    key, conf = request.form.get("model", "car"), float(request.form.get("conf", 0.25))
    files = request.files.getlist("images")
    if key not in MODELS or not files: return jsonify({"error": "Invalid request"}), 400
        
    results = []
    for file in files:
        res = run_inference_no_dispatch(key, file.read(), conf)
        if "error" not in res: results.append(res)
            
    dispatched = global_auto_dispatch()
    check_auto_completion() 
    
    clean_results = [{k: v for k, v in r.items() if k != "annotated_image"} for r in results]
    return jsonify({"inferences": clean_results, "first_image": results[0]["annotated_image"] if results else None, "dispatched_count": dispatched})

@app.route("/api/log")
def api_log():
    check_auto_completion()
    return jsonify(list(_log)[-int(request.args.get("n", 200)):])

@app.route("/api/assign_labor/<int:record_id>", methods=["POST"])
def api_assign_labor(record_id):
    data = request.json or {}
    team, status = data.get("team"), data.get("status")

    for rec in _log:
        if rec["id"] == record_id:
            if rec["priority_level"] == "NORMAL": return jsonify({"error": "No repair needed"}), 400

            if not team or team == 'null':
                rec["labor_team"], rec["labor_status"], rec["eta"], rec["arrival_time"] = None, "Unassigned", None, None
                return jsonify({"ok": True, "message": "Assignment cleared"})
            
            if status == "Completed":
                rec["labor_team"], rec["labor_status"] = team, "Completed"
                return jsonify({"ok": True, "message": f"Job marked Completed"})

            if team in TEAMS:
                rec["labor_team"], rec["labor_status"] = team, status or "Manual Reassign"
                dist_km = haversine(TEAMS[team]["lat"], TEAMS[team]["lon"], rec["gps"]["lat"], rec["gps"]["lon"])
                
                arr_iso, comp_iso = calculate_auto_eta(team, dist_km, rec["detections"])
                rec["arrival_time"] = arr_iso
                rec["eta"] = comp_iso
                
                TEAMS[team]["lat"], TEAMS[team]["lon"] = rec["gps"]["lat"], rec["gps"]["lon"]
                return jsonify({"ok": True, "message": f"Updated assignment for {team}"})

    return jsonify({"error": "Record not found"}), 404

@app.route("/api/stats")
def api_stats():
    check_auto_completion()
    records = list(_log)
    by_class, by_priority, by_model = {}, {}, {}
    
    completed = sum(1 for r in records if r.get("labor_status") == "Completed")
    yet_to_fix = sum(1 for r in records if r.get("labor_team") and r.get("labor_status") != "Completed")

    for rec in records:
        p, m = rec["priority_level"], rec["model_key"]
        by_priority[p], by_model[m] = by_priority.get(p, 0) + 1, by_model.get(m, 0) + 1
        for det in rec.get("detections", []):
            c = det["class"]
            by_class[c] = by_class.get(c, 0) + 1

    return jsonify({
        "total": len(records), "by_class": by_class, "by_priority": by_priority,
        "completed": completed, "yet_to_fix": yet_to_fix,
        "avg_score": round(sum(r["priority_score"] for r in records) / len(records), 3) if records else 0,
        "critical_pct": round(by_priority.get("CRITICAL", 0) / len(records) * 100, 1) if records else 0,
    })

@app.route("/api/sim_time_leap", methods=["POST"])
def api_sim_time_leap():
    """Developer tool: Fast forwards the system clock by X hours."""
    hours = request.json.get("hours", 2) if request.json else 2
    
    for team in TEAMS.values():
        team["available_from"] -= timedelta(hours=hours)
        
    for rec in _log:
        if rec["labor_status"] not in ["Completed", "Unassigned"]:
            if rec.get("eta"):
                dt_eta = datetime.strptime(rec["eta"], "%Y-%m-%dT%H:%M:%S.000Z") - timedelta(hours=hours)
                rec["eta"] = dt_eta.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if rec.get("arrival_time"):
                dt_arr = datetime.strptime(rec["arrival_time"], "%Y-%m-%dT%H:%M:%S.000Z") - timedelta(hours=hours)
                rec["arrival_time"] = dt_arr.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
    check_auto_completion()
    return jsonify({"ok": True, "message": f"Simulated +{hours} Hour(s). System ETAs updated."})

@app.route("/api/clear_log", methods=["POST"])
def api_clear_log():
    _log.clear()
    return jsonify({"ok": True})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    print("=" * 55)
    print("  Road Defect Monitor — Time-Optimized Routing API")
    print("  http://0.0.0.0:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)