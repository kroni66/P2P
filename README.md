# P2P Distributed Training Project

This project demonstrates a simple P2P gradient synchronization architecture using a central broker and multiple worker instances.

## Components

- **broker**: FastAPI application that collects and aggregates gradients.
- **worker**: PyTorch worker that performs a dummy training loop and syncs gradients with the broker.

## Prerequisites

- Docker
- NVIDIA Container Toolkit (for GPU support)
- docker-compose

## Usage

1. Build and start all services:

   ```bash
   docker-compose up --build
   ```

2. The broker will be available on `http://localhost:8000`.

3. Workers will start training and synchronize gradients automatically.

## Configuration

- **N_WORKERS**: Number of workers to wait for before aggregating gradients (default: 2).
- **EPOCHS**: Number of epochs for worker training (default: 5).

You can override defaults by setting environment variables:

```bash
export N_WORKERS=3
export EPOCHS=10
docker-compose up --build
```
 