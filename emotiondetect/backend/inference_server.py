# backend/inference_server.py
from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")  # allow connections from any frontend

# Load model
emotion_model = load_model('emotion_model_full.keras')

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Map to store client usernames
clients = {}

@socketio.on('register')
def register(data):
    """Register a username for a client"""
    username = data.get('username', 'Anonymous')
    clients[request.sid] = username
    print(f"[{username}] Connected.")

@socketio.on('disconnect')
def disconnect():
    username = clients.get(request.sid, 'Anonymous')
    print(f"[{username}] Disconnected.")
    clients.pop(request.sid, None)

@socketio.on('frame')
def handle_frame(data):
    """Handle incoming frame from client"""
    username = clients.get(request.sid, 'Anonymous')

    try:
        header, encoded = data.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[{username}] Failed to decode frame: {e}")
        return

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

        prediction = emotion_model.predict(cropped_img, verbose=0)[0]
        maxindex = int(np.argmax(prediction))
        label = emotion_dict[maxindex]
        confidence = float(np.max(prediction)) * 100

        # Print in terminal with client username
        print(f"[{username}] {label} ({confidence:.1f}%)")

        # Send prediction back to client
        emit('prediction', {'label': label, 'confidence': confidence})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

