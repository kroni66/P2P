# Use Python slim base image
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI broker
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]