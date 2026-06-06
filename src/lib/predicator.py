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

            confidence_a = torch.softmax(logits_a, dim=1).max(dim=1).values.mean()
            confidence_b = torch.softmax(logits_b, dim=1).max(dim=1).values.mean()

            logits_n = (confidence_a * logits_a + confidence_b * logits_b) / (confidence_a + confidence_b)

            loss = criterion(logits_n, y_batch)
            preds = logits_n.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / total

        return valid_loss, valid_accuracy

def pred_dynamic(valid_loader, model, criterion):
    constants = MetaConstants()
    device = constants.DEVICE
    use_cuda = constants.USE_CUDA

    correct = 0
    total = 0
    valid_loss = 0.0

    with torch.no_grad():

        for x_batch, y_batch in valid_loader:
            
            x_batch = x_batch.to(device, non_blocking=use_cuda)

            x_orig_bright = x_batch * 1.1
            x_orig_dark = x_batch * 0.9

            x_flipped_norm = torch.flip(x_batch, dims=[3])
            x_flipped_bright = x_flipped_norm * 1.1
            x_flipped_dark = x_flipped_norm * 0.9

            y_batch = y_batch.to(device, non_blocking=use_cuda)

            logits_a = model(x_batch)
            logits_b = model(x_orig_bright)
            logits_c = model(x_orig_dark)
            logits_d = model(x_flipped_norm)
            logits_e = model(x_flipped_bright)
            logits_f = model(x_flipped_dark)

            conf_a = torch.softmax(logits_a, dim=1).max(dim=1).values.mean()
            conf_b = torch.softmax(logits_b, dim=1).max(dim=1).values.mean()
            conf_c = torch.softmax(logits_c, dim=1).max(dim=1).values.mean()
            conf_d = torch.softmax(logits_d, dim=1).max(dim=1).values.mean()
            conf_e = torch.softmax(logits_e, dim=1).max(dim=1).values.mean()
            conf_f = torch.softmax(logits_f, dim=1).max(dim=1).values.mean()


            logits_n = (conf_a * logits_a + 
                        conf_b * logits_b + 
                        conf_c * logits_c +
                        conf_d * logits_d + 
                        conf_e * logits_e + 
                        conf_f * logits_f) / (
                            conf_a + 
                            conf_b + 
                            conf_c + 
                            conf_d + 
                            conf_e + 
                            conf_f)

            loss = criterion(logits_n, y_batch)
            preds = logits_n.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / total

        return valid_loss, valid_accuracy