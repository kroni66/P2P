import os
from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List, Dict

app = FastAPI()

# Configuration
N_WORKERS = int(os.getenv("N_WORKERS", "2"))

# In-memory stores
gradient_store: Dict[int, List[List[List[float]]]] = {}
aggregated_store: Dict[int, List[List[float]]] = {}

class GradientPayload(BaseModel):
    worker_id: str
    gradients: List[List[float]]
    epoch: int

@app.post("/sync_gradients")
def sync_gradients(payload: GradientPayload):
    epoch = payload.epoch
    grads = payload.gradients

    if epoch not in gradient_store:
        gradient_store[epoch] = []
    gradient_store[epoch].append(grads)

    if len(gradient_store[epoch]) < N_WORKERS:
        return {"status": "waiting", "received": len(gradient_store[epoch])}

    # Aggregate gradients
    collected = gradient_store.pop(epoch)
    num = len(collected)
    aggregated = []
    # Assume gradients is a list per layer
    for layer_grads in zip(*collected):
        # layer_grads is a tuple of lists from each worker
        avg_layer = [sum(vals) / num for vals in zip(*layer_grads)]
        aggregated.append(avg_layer)

    aggregated_store[epoch] = aggregated
    return {"status": "ok", "aggregated_gradients": aggregated}

@app.get("/aggregated/{epoch}")
def get_aggregated(epoch: int):
    if epoch not in aggregated_store:
        return {"status": "waiting"}
    aggregated = aggregated_store.pop(epoch)
    return {"status": "ok", "aggregated_gradients": aggregated}

@app.get("/health")
def health():
    return {"status": "ok", "workers_expected": N_WORKERS}