import os
import io
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, Response, render_template
from PIL import Image
import numpy as np

# --- 1. Application Setup ---
app = Flask(__name__)

# --- 2. Model Configuration & Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# This class list matches your Jupyter Notebook's training data
CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path):
    """
    Loads the ConvNeXt Tiny model and modifies the classifier for 6 classes.
    """
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

MODEL_PATH = 'convnext_fer2013.pth' 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please make sure it's in the same directory.")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- 3. Face Detection Setup ---
# Download the XML file from OpenCV's GitHub repository:
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file not found at {FACE_CASCADE_PATH}. Please download it.")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
print("✅ Face detector loaded successfully!")


# --- 4. Image Transformation ---
# This transform pipeline matches your notebook's `val_transform`.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_emotion(face_image):
    """
    Takes a PIL image of a face, transforms it, and returns the predicted emotion.
    """
    try:
        # The transform expects a PIL image
        tensor = val_transform(face_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            predicted_idx = top_catid.item()

        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = top_prob.item()
        
        return f"{predicted_class} ({confidence:.2f})"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"


def generate_frames():
    """
    Generator function to capture frames, detect faces, predict emotions, and yield frames for streaming.
    """
    camera = cv2.VideoCapture(0) # Use 0 for the default webcam
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        else:
            # Convert frame to grayscale for the face detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48) # Match the size of images the model was trained on
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face (on the color frame)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Crop the face region for the model
                face_roi_bgr = frame[y:y+h, x:x+w]
                
                # Convert the OpenCV BGR image to a PIL RGB image
                face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_roi_rgb)

                # Get the emotion prediction
                emotion_text = predict_emotion(pil_image)

                # Put the emotion text above the rectangle
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the format required for multipart streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


# --- 5. Flask Routes ---
@app.route('/')
def index():
    """Renders the main page with the video feed."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- 6. Main Execution ---
if __name__ == '__main__':
    # Note: Using debug=True with live video can sometimes cause issues.
    # Set to False if you experience lag or crashes.
    app.run(debug=False, host='0.0.0.0', port=5000)
