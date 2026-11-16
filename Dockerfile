# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train_model.py .
COPY app.py .

# Train the Machine Learning model
RUN python train_model.py

# Expose application port
EXPOSE 5000

# Start Flask application
CMD ["python", "app.py"]