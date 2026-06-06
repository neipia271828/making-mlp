import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from datetime import datetime
from statistics import median

from CONSTANTS import CONSTANTS

from lib.factory import build_dataloaders, build_model, load_model_constants
from lib.graph_drawer import graph_drawer
from lib.log_maker import make_summalize_csv, make_train_log
from lib.couting_time import Timer
from lib.save_model import save_best_checkpoint, save_model
from lib.predicator import pred, pred_double

def _summarize_last_epochs(values: list[float], window: int = 10) -> tuple[float, float]:
    last_values = values[-window:] if len(values) >= window else values
    return sum(last_values) / len(last_values), median(last_values)

def main() -> None:
    meta_constants = CONSTANTS
    model_constants = load_model_constants(meta_constants.MODEL)
    num_epochs = model_constants.NUM_EPOCHS
    learning_rate = model_constants.L_LATE
    weight_decay = model_constants.WEIGHT_DECAY

    timer_0 = Timer()
    timer_0.start()
    device = meta_constants.DEVICE
    print(device)
    use_cuda = meta_constants.USE_CUDA

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    train_loader, valid_loader = build_dataloaders(meta_constants.MODEL, device)

    model = build_model(meta_constants.MODEL).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None
    if getattr(model_constants, "SCHEDULER_NAME", None) == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=model_constants.ETA_MIN,
        )

    train_losses = []
    train_accuracies = []

    valid_losses = []
    valid_accuracies = []

    logs = []
    best_valid_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        timer_1 = Timer()
        timer_1.start()

        model.train()
        train_loss = 0.0
        valid_loss = 0.0

        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=use_cuda)
            y_batch = y_batch.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += y_batch.size(0)
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        train_losses.append(train_loss)

        correct = 0
        total = 0

        model.eval()

        if epoch + 1 == num_epochs:
            valid_loss, valid_accuracy = pred_double(valid_loader, model, criterion)
        else:
            valid_loss, valid_accuracy = pred(valid_loader, model, criterion)

        valid_losses.append(valid_loss)

        valid_accuracies.append(valid_accuracy)

        if valid_loss < best_valid_loss and meta_constants.BACKUP_BOUNDARY < valid_accuracy:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            save_best_checkpoint(
                meta_constants.PROJECT,
                timestamp,
                model,
                best_epoch,
                valid_loss,
                valid_accuracy,
            )

        if scheduler is not None:
            scheduler.step()

        timer_1.end()

        ep_time = timer_1.get()

        logs.append([ep_time, epoch+1, train_loss, train_accuracy, valid_loss, valid_accuracy])

        if epoch % 10 == 0 or epoch + 1 == num_epochs:
            print(f"ep={epoch+1}, ep_t={ep_time:.4f}, loss_t={train_loss:.4f}, acc_t={train_accuracy:.3f}, loss_v={valid_loss:.4f}, acc_v={valid_accuracy:.3f}")

    print(f"best checkpoint: epoch={best_epoch}, valid_loss={best_valid_loss:.4f}")

    if meta_constants.WRITE_TRAIN_LOG:
        make_train_log(meta_constants.PROJECT, timestamp, logs)

    ########## 学習結果保存 ###########
    timer_0.end()

    epochs = range(1, num_epochs + 1)

    req_time = timer_0.get()
    time_per_ep = req_time / num_epochs

    print(f"time = {req_time}")

    save_model(meta_constants.PROJECT, timestamp, model)

    train_loss_last10_avg, train_loss_last10_median = _summarize_last_epochs(train_losses)
    train_accuracy_last10_avg, train_accuracy_last10_median = _summarize_last_epochs(train_accuracies)
    valid_loss_last10_avg, valid_loss_last10_median = _summarize_last_epochs(valid_losses)
    valid_accuracy_last10_avg, valid_accuracy_last10_median = _summarize_last_epochs(valid_accuracies)

    if meta_constants.WRITE_SUMMARY_LOG:
        make_summalize_csv(
            meta_constants.PROJECT,
            meta_constants,
            model_constants,
            {
                "time_stamp": timestamp,
                "req_time": req_time,
                "ep_time": time_per_ep,
                "train_loss_last10_avg": train_loss_last10_avg,
                "train_loss_last10_median": train_loss_last10_median,
                "train_accuracy_last10_avg": train_accuracy_last10_avg,
                "train_accuracy_last10_median": train_accuracy_last10_median,
                "valid_loss_last10_avg": valid_loss_last10_avg,
                "valid_loss_last10_median": valid_loss_last10_median,
                "valid_accuracy_last10_avg": valid_accuracy_last10_avg,
                "valid_accuracy_last10_median": valid_accuracy_last10_median,
            },
        )

    graph_drawer(
        meta_constants.PROJECT,
        epochs,
        timestamp,
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        meta_constants.DRAW_TRAIN_GRAPH
    )


if __name__ == "__main__":
    main()
