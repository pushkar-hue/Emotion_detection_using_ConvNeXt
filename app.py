import os
import io
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, Response, render_template, request
from PIL import Image
import numpy as np
import requests
from tqdm import tqdm

# --- 1. Application Setup ---
app = Flask(__name__)

# --- 2. Configuration and Asset Management ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- File Paths and URLs ---
# Use a subdirectory for assets to make volume mounting easier on Railway
ASSETS_DIR = "assets"
MODEL_PATH = os.path.join(ASSETS_DIR, "convnext_fer2013.pth")
FACE_CASCADE_PATH = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

# --- IMPORTANT: Create a GitHub Release and paste the direct download URLs here ---
MODEL_URL = "sha256:4020519b1b8394c05b56fec8c9cb8011effd8480af7df863efb7a3b6df851cb5"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def download_file(url, path):
    """Downloads a file from a URL to a path if it doesn't exist, with a progress bar."""
    if os.path.exists(path):
        print(f"File already exists at {path}. Skipping download.")
        return
    
    print(f"File not found at {path}. Downloading from {url}...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(path)
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
                
        if total_size != 0 and progress_bar.n != total_size:
            print("Error: Download incomplete.")
            os.remove(path) # Clean up incomplete file
        else:
            print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(path):
            os.remove(path) # Clean up incomplete file

# --- 3. Model Definition ---
def load_model(model_path):
    """Loads the ConvNeXt Tiny model and modifies the classifier."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    return model

# --- 4. Image Transformation ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_emotion(face_image):
    """Predicts emotion from a PIL image of a face."""
    try:
        tensor = val_transform(face_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            predicted_class = CLASS_NAMES[top_catid.item()]
            confidence = top_prob.item()
        return f"{predicted_class} ({confidence:.2f})"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"

def generate_frames():
    """Generator for video streaming."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi_bgr = frame[y:y+h, x:x+w]
                face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_roi_rgb)
                emotion_text = predict_emotion(pil_image)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

# --- 5. Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 6. Main Execution ---
if __name__ == '__main__':
    # Step 1: Download assets if they don't exist
    download_file(MODEL_URL, MODEL_PATH)
    download_file(CASCADE_URL, FACE_CASCADE_PATH)

    # Step 2: Load models now that files are guaranteed to be present
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    print("Loading face detector...")
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    print("✅ Face detector loaded successfully!")

    # Step 3: Start the web server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


