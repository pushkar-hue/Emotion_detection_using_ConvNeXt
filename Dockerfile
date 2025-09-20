# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# This is done in a separate step to leverage Docker's layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port the app runs on

EXPOSE 5000

# Run the Flask app
# The command directly runs the python script.
# The FLASK_* environment variables are not needed for this command.
CMD ["python", "app.py"]
