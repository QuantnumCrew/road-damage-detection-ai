<!DOCTYPE html>
<html>
<head>
    <title>AI Road Damage Detection</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">

    <h1>🚧 AI Road Damage Detection System</h1>

    <h2>📌 Overview</h2>
    <p>
        This project is an AI-powered system that detects road damages such as 
        <b>potholes, cracks, and manholes</b> using a YOLO model.
        It integrates real-time detection, web interface, and automation alerts.
    </p>

    <h2>🎯 Problem Statement</h2>
    <ul>
        <li>Accidents due to poor road conditions 🚑</li>
        <li>Traffic delays 🚗</li>
        <li>Manual inspection is slow ❌</li>
    </ul>

    <h2>💡 Solution</h2>
    <pre>
User → Frontend → Flask → YOLO → n8n → Email Alert 🚨
    </pre>

    <h2>🧠 Features</h2>
    <ul>
        <li>Detect potholes, cracks, and manholes</li>
        <li>Image upload detection 📷</li>
        <li>Webcam-based real-time detection 🎥</li>
        <li>Severity classification (Low / Medium / High)</li>
        <li>Automated alerts using n8n 📩</li>
    </ul>

    <h2>🧰 Tech Stack</h2>
    <table border="1" cellpadding="8">
        <tr>
            <th>Layer</th>
            <th>Technology</th>
        </tr>
        <tr>
            <td>Frontend</td>
            <td>HTML, CSS, JavaScript</td>
        </tr>
        <tr>
            <td>Backend</td>
            <td>Flask</td>
        </tr>
        <tr>
            <td>AI Model</td>
            <td>YOLOv8</td>
        </tr>
        <tr>
            <td>Automation</td>
            <td>n8n</td>
        </tr>
    </table>

    <h2>📂 Project Structure</h2>
    <pre>
road-ai-project/
│
├── backend/
├── frontend/
├── dataset/ (excluded)
├── README.html
    </pre>

    <h2>⚙️ Setup</h2>
    <pre>
cd backend
pip install -r requirements.txt
python app.py
    </pre>

    <h2>🔄 API</h2>
    <pre>
POST /detect

Response:
{
  "type": "pothole",
  "severity": "High"
}
    </pre>

    <h2>🔔 Automation</h2>
    <p>
        Workflow: Webhook → IF (High Severity) → Email Alert 🚨
    </p>

    <h2>🚀 Future Scope</h2>
    <ul>
        <li>GPS-based detection 📍</li>
        <li>Municipality integration 🏛️</li>
        <li>Analytics dashboard 📊</li>
        <li>Mobile app 📱</li>
    </ul>

    <h2>🏁 Conclusion</h2>
    <p>
        This project shows how AI + Automation can improve road safety and smart city infrastructure.
    </p>

    <h3>⭐ Give this project a star on GitHub!</h3>

</body>
</html>
