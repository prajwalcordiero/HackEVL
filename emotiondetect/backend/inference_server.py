from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model

# Load the fixed model
model = load_model("emotion_model_full_fixed.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get image from request
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    # Preprocess for model
    face = cv2.resize(frame, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Predict
    preds = model.predict(face, verbose=0)
    emotion = emotion_labels[np.argmax(preds)]

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
