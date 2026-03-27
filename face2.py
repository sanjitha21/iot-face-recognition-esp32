import cv2
import numpy as np
import os
import urllib.request
from deepface import DeepFace
import pyttsx3
import threading
from flask import Flask, render_template_string, Response
import time
from datetime import datetime

# ---------------- CONFIG ----------------
ESP32_URLS = [
    "http://10.235.211.44/cam-hi.jpg",  # primary
    "http://10.235.211.44/cam-lo.jpg"   # fallback
]
TRAINING_PATH = r"C:\Users\Sanjitha\OneDrive\Pictures\Documents\espface_(3)[1]\espface (2)\espface\espface\training_images"
THRESHOLD = 0.55
SPEAK_INTERVAL = 5  # seconds between greetings for same person

# ---------------- SPEECH ----------------
def speak_text(message):
    def run_tts(msg):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        preferred = 1 if len(voices) > 1 else 0
        engine.setProperty("voice", voices[preferred].id)
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.say(msg)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=run_tts, args=(message,), daemon=True).start()

# ---------------- HELPERS ----------------
def normalize(v):
    v = np.array(v).reshape(-1)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(v1, v2):
    v1 = np.array(v1).reshape(-1)
    v2 = np.array(v2).reshape(-1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ---------------- LOAD TRAINING DATA ----------------
print("Loading known faces...")
known_embeddings = {}
for person_name in os.listdir(TRAINING_PATH):
    person_folder = os.path.join(TRAINING_PATH, person_name)
    if not os.path.isdir(person_folder):
        continue

    embeddings_list = []
    for img_file in os.listdir(person_folder):
        file_path = os.path.join(person_folder, img_file)
        try:
            emb = DeepFace.represent(file_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
            embeddings_list.append(normalize(emb))
            print(f"✅ Loaded {img_file} for {person_name}")
        except Exception as e:
            print(f"⚠️ Skipping {img_file}: {e}")

    if embeddings_list:
        known_embeddings[person_name] = embeddings_list

print(f"✅ Training loaded for {len(known_embeddings)} people.\n")

# ---------------- FLASK APP ----------------
app = Flask(__name__)
greeted_people = {}
stop_camera = False  # flag to quit camera stream
greeted_on_open = False  # one-time greeting when camera first opens

# HTML template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>ESP32-CAM Face Recognition</title>
  <style>
    body { background: #111; color: #eee; text-align: center; font-family: Arial; }
    img { width: 80%; border-radius: 10px; margin-top: 20px; box-shadow: 0 0 15px #0f0; }
    h1 { color: #0f0; }
  </style>
</head>
<body>
  <h1>ESP32-CAM Face Recognition</h1>
  <img src="{{ url_for('video_feed') }}">
  <p>Press <b>Q</b> in the terminal to quit camera</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ---------------- CAMERA FEED ----------------
def get_frame():
    """Try to read from ESP32-CAM (hi -> lo)"""
    for url in ESP32_URLS:
        try:
            img_resp = urllib.request.urlopen(url, timeout=5)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is not None:
                return frame
        except Exception as e:
            print(f"⚠️ Connection failed: {url} ({e})")
    return None

def gen_frames():
    """Continuously capture from ESP32-CAM and yield processed frames"""
    global stop_camera, greeted_people
    while not stop_camera:
        frame = get_frame()
        if frame is None:
            time.sleep(1)
            continue

        # One-time greeting when camera feed opens
        global greeted_on_open
        if not greeted_on_open:
            hour = datetime.now().hour
            # Simple split: before 12 -> morning, 12-17 -> afternoon, after 17 -> evening
            if hour < 12:
                greeting = "Good morning"
            elif hour < 17:
                greeting = "Good afternoon"
            else:
                greeting = "Good evening"
            print(f"🕒 {hour}:00 - {greeting}")
            speak_text(greeting)
            # also greet everyone as requested
            print("Hi everyone")
            speak_text("Hi everyone")
            greeted_on_open = True

        try:
            detections = DeepFace.extract_faces(frame, detector_backend="opencv")
        except Exception as e:
            print("Face extraction error:", e)
            detections = []

        for face in detections:
            x, y, w, h = (face["facial_area"][k] for k in ("x", "y", "w", "h"))
            face_img = face["face"]

            try:
                embedding = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                embedding = normalize(embedding)
            except:
                continue

            name = "Unknown"
            max_sim = -1
            for person_name, embeddings_list in known_embeddings.items():
                for known_emb in embeddings_list:
                    sim = cosine_similarity(embedding, known_emb)
                    if sim > max_sim and sim > (1 - THRESHOLD):
                        max_sim = sim
                        name = person_name

            # Time-based speaking
            current_time = time.time()
            if name != "Unknown":
                last_time = greeted_people.get(name, 0)
                if current_time - last_time > SPEAK_INTERVAL:
                    current_datetime = datetime.now()
                    current_time_str = current_datetime.strftime("%I:%M %p")  # Format: HH:MM AM/PM
                    print(f"👋 Hi {name}, the time is {current_time_str}")
                    speak_text(f"Hi {name}, the time is {current_time_str}")
                    greeted_people[name] = current_time

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x + 5, y + h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        # show locally (press Q to quit)
        cv2.imshow("ESP32-CAM Feed (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quit requested. Closing camera...")
            stop_camera = True
            break

        # stream to web browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- RUN ----------------
if __name__ == '__main__':
    print("🚀 Flask server running at: http://127.0.0.1:5000")
    print("🎥 Press 'Q' in the console to quit camera anytime.")
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        stop_camera = True
        cv2.destroyAllWindows()
        print("🛑 Server stopped manually.")
