import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from datetime import datetime
from pathlib import Path
import csv

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten_tensor),
    ])


def build_dataloaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = build_transform()

    train_ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    valid_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    return train_loader, valid_loader


PROJECT = "FashionMNIST"
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    device = get_device()
    print(device)

    train_loader, valid_loader = build_dataloaders(device)
    use_cuda = device.type == "cuda"

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1
    train_losses = []
    valid_accuracies = []

    logs_dir = Path(f"data/{PROJECT}/logs") / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    logs_path = logs_dir / "train_logs.csv"

    with open(logs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ep_time", "epoch", "train_loss", "valid_accuracy"])

        for epoch in range(num_epochs):
            ep_time_st = time.perf_counter()
            model.train()
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=use_cuda)
                y_batch = y_batch.to(device, non_blocking=use_cuda)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_losses.append(total_loss)

            correct = 0
            total = 0

            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    x_batch = x_batch.to(device, non_blocking=use_cuda)
                    y_batch = y_batch.to(device, non_blocking=use_cuda)

                    logits = model(x_batch)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)

            accuracy = correct / total
            valid_accuracies.append(accuracy)

            ep_time_ed = time.perf_counter()

            ep_time = ep_time_ed - ep_time_st

            writer.writerow([ep_time, epoch+1, total_loss, accuracy])
            print(f"ep_time={ep_time:.4f}, epoch={epoch+1}, loss={total_loss:.4f}, accuracy={accuracy:.3f}")

    ########## 学習結果保存 ###########
    end = time.perf_counter()

    epochs = range(1, num_epochs + 1)

    req_time = end - start
    time_per_ep = req_time / num_epochs

    print(f"time = {req_time}")

    model_dir = Path(f"data/{PROJECT}/models") / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pt"

    torch.save(model.state_dict(), model_path)

    import matplotlib.pyplot as plt
    csv_path = Path(".") / f"data/{PROJECT}/logs/{PROJECT}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(["time_stamp", "req_time", "ep_time", "epochs", "total_loss", "accuracy"])
        writer.writerow([timestamp, req_time, time_per_ep, num_epochs, total_loss, accuracy])

    graph_dir = Path(f"data/{PROJECT}/logs") / timestamp
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / "train_graph.svg"

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_accuracies, marker="o")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.show()


if __name__ == "__main__":
    main()
