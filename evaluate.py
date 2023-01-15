import torch
import tqdm
import numpy as np

def evaluate(model, data_loader, criterion, device):
    
    model.eval()    
    accuracy, num_corrects, num_samples = 0, 0, 0
    loss, val_loss = 0, 0
    model.zero_grad(set_to_none=True)

    for imgs, labels in data_loader:
            
        imgs = imgs.to(device, dtype=torch.float)
        labels = labels.type(torch.LongTensor)

        labels = labels.to(device)
        with torch.set_grad_enabled(False):

            outputs = model(imgs)
            _, yhats = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            num_corrects += torch.sum(yhats == labels.data)
            num_samples += yhats.size(0)
    accuracy = float(100 * num_corrects / num_samples)
    val_loss = float(val_loss)
    # print(f'Epoch: {epoch} | Accuracy: {accuracy:.2f}%')
    
    return accuracy, val_loss

