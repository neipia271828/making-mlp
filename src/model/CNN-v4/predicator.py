import torch
from CONSTANTS import MetaConstants

def pred(valid_loader, model, criterion):
    constants = MetaConstants()

    device = constants.DEVICE
    use_cuda = constants.USE_CUDA

    correct = 0
    total = 0
    valid_loss = 0.0

    with torch.no_grad():

        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device, non_blocking=use_cuda)
            y_batch = y_batch.to(device, non_blocking=use_cuda)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / total

        return valid_loss, valid_accuracy

def pred_double(valid_loader, model, criterion):
    constants = MetaConstants()

    device = constants.DEVICE
    use_cuda = constants.USE_CUDA

    correct = 0
    total = 0
    valid_loss = 0.0

    with torch.no_grad():

        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device, non_blocking=use_cuda)
            x_flipped = torch.flip(x_batch, dims=[3])

            y_batch = y_batch.to(device, non_blocking=use_cuda)

            logits_a = model(x_batch)
            logits_b = model(x_flipped)

            conf_a = torch.softmax(logits_a, dim=1).max(dim=1).values.mean()
            conf_b = torch.softmax(logits_b, dim=1).max(dim=1).values.mean()

            logits_n = (conf_a * logits_a + conf_b * logits_b) / (conf_a + conf_b)

            loss = criterion(logits_n, y_batch)
            preds = logits_n.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / total

        return valid_loss, valid_accuracy
