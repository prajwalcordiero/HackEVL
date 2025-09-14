import eventlet
eventlet.monkey_patch()
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model


# Flask setup
app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load model
emotion_model = load_model("emotion_model_full.keras")


# Emotion dictionary
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Classify emotions
ACTIVE_EMOTIONS = {"Happy", "Sad", "Neutral", "Angry", "Disgusted"}
INACTIVE_EMOTIONS = {"Fearful", "Surprised"}

# Inactivity timeout
INACTIVE_TIMEOUT = 300  # seconds (adjust to 600 for production)

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Client state
clients = {}          # sid -> username
user_status = {}      # sid -> "active"/"inactive"
last_active = {}      # sid -> timestamp


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@socketio.on("register")
def register(data):
    username = data.get("username", "Anonymous")
    sid = request.sid
    clients[sid] = username
    user_status[sid] = "active"
    last_active[sid] = time.time()
    print(f"[{username}] Connected.")

    # Emit initial status
    emit("status", {"username": username, "status": "active"}, broadcast=True)


@socketio.on("disconnect")
def disconnect():
    sid = request.sid
    username = clients.get(sid, "Anonymous")
    print(f"[{username}] Disconnected.")
    clients.pop(sid, None)
    user_status.pop(sid, None)
    last_active.pop(sid, None)
    emit("status", {"username": username, "status": "offline"}, broadcast=True)


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    username = clients.get(sid, "Anonymous")

    try:
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[{username}] Frame decode failed: {e}")
        return

    label = None
    confidence = 0.0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

        prediction = emotion_model.predict(cropped_img, verbose=0)[0]
        maxindex = int(np.argmax(prediction))
        label = emotion_dict[maxindex]
        confidence = float(np.max(prediction)) * 100

        print(f"[{username}] {label} ({confidence:.1f}%)")

        # Update activity status
        if label in ACTIVE_EMOTIONS:
            last_active[sid] = time.time()
            if user_status.get(sid) != "active":
                user_status[sid] = "active"
                emit("chat", {"username": "System", "message": f"{username} is now active."}, broadcast=True)
                emit("status", {"username": username, "status": "active"}, broadcast=True)

        elif label in INACTIVE_EMOTIONS:
            # Mark as inactive immediately
            if user_status.get(sid) != "inactive":
                user_status[sid] = "inactive"
                emit("chat", {"username": "System", "message": f"{username} is inactive (emotion: {label})."}, broadcast=True)
                emit("status", {"username": username, "status": "inactive"}, broadcast=True)

    # If no face detected
    else:
        print(f"[{username}] No face detected.")
        # Don't update last_active; timeout thread will handle it

    # Send back prediction to client
    emit("prediction", {
        "label": label if label else "No Face",
        "confidence": confidence,
    })


@socketio.on("chat")
def handle_chat(data):
    username = clients.get(request.sid, "Anonymous")
    message = data.get("message", "")
    print(f"[CHAT] {username}: {message}")
    emit("chat", {"username": username, "message": message}, broadcast=True)


# Background thread to check inactivity
def check_inactivity_loop():
    while True:
        now = time.time()
        for sid in list(clients.keys()):
            status = user_status.get(sid, "active")
            last = last_active.get(sid, now)
            elapsed = now - last
            if elapsed > INACTIVE_TIMEOUT and status != "inactive":
                user_status[sid] = "inactive"
                username = clients.get(sid, "Anonymous")
                print(f"[{username}] Auto-set to INACTIVE after {elapsed:.1f}s of inactivity.")
                socketio.emit("chat", {"username": "System", "message": f"{username} is inactive due to inactivity."}, broadcast=True)
                socketio.emit("status", {"username": username, "status": "inactive"}, broadcast=True)
        time.sleep(3)  # check every 3 seconds


# Start background thread
threading.Thread(target=check_inactivity_loop, daemon=True).start()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
