from ultralytics import YOLO
import cv2

# Load model once
model = YOLO("backend/model/best.pt")

def detect_damage(image_path):
    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            detections.append({
                "type": label,
                "confidence": round(conf, 2)
            })

    if detections:
        # Take first detection
        damage = detections[0]["type"]

        # Simple severity logic
        if damage == "pothole":
            severity = "High"
        elif damage == "crack":
            severity = "Medium"
        else:
            severity = "Low"

        return {
            "type": damage,
            "severity": severity
        }

    return {
        "type": "No Damage",
        "severity": "None"
    }
