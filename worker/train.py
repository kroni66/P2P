import os
import time
import requests  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

# Configuration
BROKER_URL = os.getenv("BROKER_URL", "http://broker:8000")
WORKER_ID = os.getenv("WORKER_ID", "worker")
N_WORKERS = int(os.getenv("N_WORKERS", "2"))
EPOCHS = int(os.getenv("EPOCHS", "5"))

# Dummy model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def get_aggregated(epoch):
    url = f"{BROKER_URL}/aggregated/{epoch}"
    resp = requests.get(url)
    return resp.json()

def main():
    model = Model().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        # Dummy data
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        # Extract gradients
        gradients = [param.grad.view(-1).tolist() for param in model.parameters()]

        # Send gradients to broker
        payload = {"worker_id": WORKER_ID, "gradients": gradients, "epoch": epoch}
        resp = requests.post(f"{BROKER_URL}/sync_gradients", json=payload)
        data = resp.json()

        if data.get("status") == "waiting":
            print(f"[{WORKER_ID}] Waiting for other workers... ({data.get('received')}/{N_WORKERS})")
            # Poll for aggregation
            while True:
                time.sleep(1)
                agg_data = get_aggregated(epoch)
                if agg_data.get("status") == "ok":
                    data = agg_data
                    break
                print(f"[{WORKER_ID}] Still waiting for aggregated gradients...")
        else:
            print(f"[{WORKER_ID}] Received aggregated gradients directly.")

        # Apply aggregated gradients
        aggregated = data["aggregated_gradients"]
        for param, agg in zip(model.parameters(), aggregated):
            param.grad = torch.tensor(agg, device='cuda').view_as(param.data)

        optimizer.step()
        print(f"[{WORKER_ID}] Completed epoch {epoch}")

if __name__ == "__main__":
    main()