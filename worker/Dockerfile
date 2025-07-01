# Use NVIDIA CUDA toolkit devel image with cuDNN
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Install Python and build tools
RUN apt-get update && apt-get install -y python3-pip python3 build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run training script
CMD ["python3", "train.py"]