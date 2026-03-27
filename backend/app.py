"""
RoadScan AI — Flask Backend
Road Damage Detection & Reporting System
Integrations: Supabase (DB + Storage) · n8n Webhooks · YOLOv8
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, json
from datetime import datetime
from pathlib import Path

# ── Optional heavy deps (graceful fallback) ──
try:
    from ultralytics import YOLO
    import cv2
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics/cv2 not installed — running mock detection")

try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARN] supabase-py not installed — Supabase calls skipped")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    import urllib.request, urllib.error
    HTTPX_AVAILABLE = False

import sqlite3  # always-on local fallback

# ════════════════════════════════════════════
#  ⚙️  CONFIG
# ════════════════════════════════════════════
SUPABASE_URL    = "https://pjlwnrbsrphaokimeuqw.supabase.co"
SUPABASE_KEY    = "sb_publishable_GrE3E-zYTyzrKtMBxZjMvw_MSfcomdm"
N8N_WEBHOOK_URL = "https://jayashree2k007.app.n8n.cloud/webhook/road-damage"

MODEL_PATH      = "best.pt"
FALLBACK_MODEL  = "yolov8n.pt"
UPLOAD_DIR      = Path("uploads")
DB_PATH         = "roadscan.db"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app, origins="*")

# ════════════════════════════════════════════
#  🔌  SUPABASE CLIENT
# ════════════════════════════════════════════
sb = None

if SUPABASE_AVAILABLE:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[INFO] Supabase client connected")
    except Exception as e:
        print(f"[WARN] Supabase init failed: {e}")


def sb_insert(table, payload):
    if not sb:
        return None
    try:
        res = sb.table(table).insert(payload).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[Supabase] Insert to '{table}' failed: {e}")
        return None


def sb_select(table, limit=50):
    if not sb:
        return []
    try:
        res = sb.table(table).select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        print(f"[Supabase] Select from '{table}' failed: {e}")
        return []


def sb_upload_image(file_bytes, filename, folder="detections"):
    if not sb:
        return None
    try:
        path = f"{folder}/{filename}"
        sb.storage.from_("road-images").upload(
            path, file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        return sb.storage.from_("road-images").get_public_url(path)
    except Exception as e:
        print(f"[Supabase Storage] Upload failed: {e}")
        return None


# ════════════════════════════════════════════
#  🔔  n8n WEBHOOK
# ════════════════════════════════════════════
def trigger_n8n(payload):
    try:
        if HTTPX_AVAILABLE:
            r = httpx.post(N8N_WEBHOOK_URL, json=payload, timeout=8)
            ok = r.status_code < 300
        else:
            body = json.dumps(payload).encode()
            req  = urllib.request.Request(
                N8N_WEBHOOK_URL, data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                ok = r.status < 300
        print(f"[n8n] {'✅' if ok else '⚠️'} status fired")
        return ok
    except Exception as e:
        print(f"[n8n] Webhook failed: {e}")
        return False


# ════════════════════════════════════════════
#  🗄️  SQLITE FALLBACK
# ════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS detections (
        id TEXT PRIMARY KEY, image_path TEXT, image_url TEXT,
        damage_types TEXT, severities TEXT, confidences TEXT,
        count INTEGER, timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY, damage_type TEXT, severity TEXT,
        location TEXT, description TEXT, reporter_name TEXT,
        contact TEXT, image_url TEXT, confidence TEXT,
        status TEXT DEFAULT 'pending', timestamp TEXT)""")
    conn.commit(); conn.close()


def db_insert_detection(row):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT OR REPLACE INTO detections
            (id,image_path,image_url,damage_types,severities,confidences,count,timestamp)
            VALUES (?,?,?,?,?,?,?,?)""",
            (row['id'], row.get('image_path',''), row.get('image_url',''),
             row['damage_types'], row['severities'], row['confidences'],
             row['count'], row['timestamp']))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[SQLite] {e}")


def db_insert_report(row):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT OR REPLACE INTO reports
            (id,damage_type,severity,location,description,reporter_name,
             contact,image_url,confidence,status,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (row['id'], row['damage_type'], row.get('severity','medium'),
             row['location'], row['description'], row.get('reporter_name',''),
             row.get('contact',''), row.get('image_url',''),
             row.get('confidence',''), 'pending', row['timestamp']))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[SQLite] {e}")


# ════════════════════════════════════════════
#  🧠  SEVERITY + YOLO
# ════════════════════════════════════════════
def compute_severity(dtype, confidence, bbox):
    area = 0
    if bbox and len(bbox) == 4:
        area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
    t = dtype.lower()
    if "pothole" in t:
        return "HIGH" if area > 50_000 or confidence > 0.90 else "MEDIUM" if area > 20_000 or confidence > 0.70 else "LOW"
    if "crack" in t:
        return "HIGH" if confidence > 0.85 else "MEDIUM" if confidence > 0.65 else "LOW"
    if "manhole" in t:
        return "HIGH" if area > 40_000 else "MEDIUM"
    return "HIGH" if confidence > 0.80 else "MEDIUM" if confidence > 0.60 else "LOW"


model = None
CLASS_MAP = {0: "Pothole", 1: "Surface Crack", 2: "Manhole"}


def load_model():
    global model
    if not YOLO_AVAILABLE:
        return
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    print(f"[INFO] Loading: {path}")
    model = YOLO(path)
    print("[INFO] Model ready ✅")


def run_yolo(image_path):
    if not model:
        return mock_detect()
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    dets = []
    for r in model(img, conf=0.35):
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            bbox   = [int(x) for x in box.xyxy[0].tolist()]
            dtype  = CLASS_MAP.get(cls_id, f"Unknown-{cls_id}")
            dets.append({"type": dtype, "confidence": round(conf, 3),
                         "severity": compute_severity(dtype, conf, bbox), "bbox": bbox})
    return dets


def mock_detect():
    import random
    opts = [("Pothole",[120,80,340,260]),("Surface Crack",[300,50,610,130]),("Manhole",[200,150,380,330])]
    chosen = random.sample(opts, random.randint(1,2))
    return [{"type":t,"confidence":round(random.uniform(0.72,0.97),3),
             "severity":compute_severity(t,0.85,b),"bbox":b} for t,b in chosen]


# ════════════════════════════════════════════
#  🌐  ROUTES
# ════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    return jsonify({"service":"RoadScan AI","status":"running",
                    "supabase_online": sb is not None,
                    "yolo_available": YOLO_AVAILABLE,
                    "model_loaded": model is not None,
                    "n8n_webhook": N8N_WEBHOOK_URL})


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' in form-data"}), 400
    file  = request.files["image"]
    ext   = Path(file.filename or "img.jpg").suffix or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = UPLOAD_DIR / fname
    file.save(str(fpath))

    # Upload to Supabase Storage
    with open(fpath, "rb") as f:
        image_url = sb_upload_image(f.read(), fname, "detections")

    detections = run_yolo(str(fpath))
    ts = datetime.utcnow().isoformat()
    det_id = uuid.uuid4().hex[:10].upper()

    sb_rec = sb_insert("detections", {
        "image_url": image_url,
        "damage_types": json.dumps([d["type"] for d in detections]),
        "severities":   json.dumps([d["severity"] for d in detections]),
        "confidences":  json.dumps([d["confidence"] for d in detections]),
        "count": len(detections),
    })
    db_insert_detection({"id":det_id,"image_path":str(fpath),"image_url":image_url or "",
        "damage_types":json.dumps([d["type"] for d in detections]),
        "severities":json.dumps([d["severity"] for d in detections]),
        "confidences":json.dumps([d["confidence"] for d in detections]),
        "count":len(detections),"timestamp":ts})

    return jsonify({
        "detection_id": sb_rec["id"] if sb_rec else det_id,
        "image_url": image_url or f"/uploads/{fname}",
        "detections": detections,
        "count": len(detections),
        "timestamp": ts,
        "saved_to": "supabase" if sb_rec else "sqlite",
    })


@app.route("/report", methods=["POST"])
def create_report():
    data = request.get_json(silent=True) or {}
    for f in ("damage_type","location","description"):
        if not data.get(f):
            return jsonify({"error": f"Missing: {f}"}), 400

    ts  = datetime.utcnow().isoformat()
    rid = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    payload = {
        "damage_type":   data["damage_type"],
        "severity":      data.get("severity","medium"),
        "location":      data["location"],
        "description":   data["description"],
        "reporter_name": data.get("reporter_name") or data.get("name","Anonymous"),
        "contact":       data.get("contact",""),
        "image_url":     data.get("image_url",""),
        "confidence":    data.get("confidence",""),
        "status":        "pending",
    }

    sb_rec = sb_insert("reports", payload)
    db_insert_report({**payload, "id": rid, "timestamp": ts})
    final_id = sb_rec["id"] if sb_rec else rid

    n8n_ok = trigger_n8n({
        "report_id":   final_id,
        "damage_type": payload["damage_type"],
        "severity":    payload["severity"],
        "location":    payload["location"],
        "description": payload["description"],
        "reporter":    payload["reporter_name"],
        "contact":     payload["contact"],
        "image_url":   payload["image_url"],
        "confidence":  payload["confidence"],
        "timestamp":   ts,
        "source":      "RoadScan AI Flask Backend",
    })

    return jsonify({"success":True,"report_id":final_id,"status":"pending",
                    "saved_to":"supabase" if sb_rec else "sqlite",
                    "n8n_fired":n8n_ok,"timestamp":ts,
                    "message":"Report saved. Authorities notified via n8n."}), 201


@app.route("/reports", methods=["GET"])
def get_reports():
    rows = sb_select("reports", 50)
    if not rows:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM reports ORDER BY timestamp DESC LIMIT 50").fetchall()]
        conn.close()
    return jsonify(rows)


@app.route("/reports/<report_id>", methods=["PATCH"])
def update_report(report_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in ["pending","reviewing","resolved","rejected"]:
        return jsonify({"error":"invalid status"}), 400
    if sb:
        try: sb.table("reports").update({"status":status}).eq("id",report_id).execute()
        except Exception as e: print(e)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))
        conn.commit(); conn.close()
    trigger_n8n({"event":"status_update","report_id":report_id,"status":status,
                 "timestamp":datetime.utcnow().isoformat()})
    return jsonify({"success":True,"report_id":report_id,"status":status})


@app.route("/stats", methods=["GET"])
def stats():
    det_rows = sb_select("detections", 500)
    rep_rows = sb_select("reports", 500)
    if not det_rows:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        det_rows = [dict(r) for r in conn.execute("SELECT * FROM detections").fetchall()]
        rep_rows = [dict(r) for r in conn.execute("SELECT * FROM reports").fetchall()]
        conn.close()
    tc = {"pothole":0,"crack":0,"manhole":0}
    for row in det_rows:
        try:
            for t in json.loads(row.get("damage_types","[]")):
                tl = t.lower()
                if "pothole" in tl: tc["pothole"]+=1
                elif "crack" in tl: tc["crack"]+=1
                elif "manhole" in tl: tc["manhole"]+=1
        except: pass
    return jsonify({
        "total_detections": len(det_rows), "total_reports": len(rep_rows),
        "pending_reports":  sum(1 for r in rep_rows if r.get("status")=="pending"),
        "resolved_reports": sum(1 for r in rep_rows if r.get("status")=="resolved"),
        "damage_breakdown": tc, "supabase_online": sb is not None,
    })


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    init_db(); load_model()
    print("\n" + "═"*54)
    print("  🛣️  RoadScan AI Backend  →  http://localhost:5000")
    print(f"  Supabase : {'✅ connected'   if sb    else '❌ SQLite fallback'}")
    print(f"  YOLO     : {'✅ loaded'      if model else ('⚠️ mock' if not YOLO_AVAILABLE else '⏳ loading')}")
    print(f"  n8n      : {N8N_WEBHOOK_URL}")
    print("═"*54 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
