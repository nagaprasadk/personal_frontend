# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (Streamlit default)
EXPOSE 80

# Run Streamlit app on container start
CMD ["streamlit", "run", "frontend.py", "--server.port=80", "--server.enableCORS=false", "--server.headless=true"]
