import sys
import os
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]

import torch
import torch.nn as nn

from CONSTANTS import CONSTANTS

from lib.factory import build_dataloaders, build_model, load_model_constants
from lib.couting_time import Timer
from lib.predicator import pred_dynamic

def main() -> None:
    meta_constants = CONSTANTS
    model_constants = load_model_constants(meta_constants.MODEL)

    timer_0 = Timer()
    timer_0.start()
    device = meta_constants.DEVICE
    print(device)

    _, valid_loader = build_dataloaders(meta_constants.MODEL, device)

    model = build_model(meta_constants.MODEL).to(device)
    state_dict = torch.load(
        "data/CIFAR10/models/2026-06-05-09-17/best_checkpoint.pt",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    valid_loss, valid_accuracy = pred_dynamic(valid_loader, model, criterion)

    print(f"valid_loss = {valid_loss:.4f}, valid_acc = {valid_accuracy:.4f}")

if __name__ == "__main__":
    main()
