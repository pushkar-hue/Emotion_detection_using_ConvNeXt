# Use official Python image as base
FROM python:3.10-slim

# Install system packages needed for Git LFS
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ----> CHANGE 1: Clone the repository instead of copying files <----
# This brings in the .git folder, which is required for git lfs pull
RUN git clone https://github.com/mohitsharmas97/Emotion_detection_using_ConvNeXt.git .

# ----> CHANGE 2: Now that it's a real git repo, pull the large files <----
RUN git lfs pull

# Install Python dependencies from the cloned repo
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
