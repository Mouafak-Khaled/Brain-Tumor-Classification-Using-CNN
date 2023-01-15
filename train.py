from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm 
import torch
import numpy as np
import os
import time
from evaluate import evaluate

default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def train(model, optimizer, train_loader, validation_loader, criterion, epochs,
          save_params=False, verbose=False, load_model=False, scheduler=None, device=None):
            

    print(f'Current device: {device}')
    
    epoch_times = []
    total_loss = []    
    model.to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    accuracies_validation = []
    accuracies_train = []
    steps = 10
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_tic = time.time()
        train_acc, num_corrects, num_samples = 0, 0, 0
        model.zero_grad(set_to_none=True)

        for img, label in tqdm(train_loader):

            img = img.to(device, non_blocking=True)
            label = label.type(torch.LongTensor).to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(img)
                loss = criterion(outputs, label)
                epoch_loss += loss.item()
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
            _, yhats = torch.max(outputs, 1)
            num_corrects += torch.sum(yhats == label.data).item()
            num_samples += yhats.size(0)
        train_acc = 100 * num_corrects / num_samples
        print('Evaluating epoch...', flush=True)

        val_acc, val_loss = evaluate(model, validation_loader, criterion, device)

        accuracies_train.append(train_acc)
        accuracies_validation.append(val_acc)
        total_loss.append(epoch_loss)        
      
        
        if scheduler != None:
            # if isinstance(scheduler, ('ReduceLROnPlateau')):
            #     scheduler.step(val_loss) 
            # else:
            scheduler.step() 
                
        
        epoch_toc = time.time()
        epoch_time = epoch_toc - epoch_tic
        epoch_times.append(epoch_time)
        epoch_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch} | Loss: {epoch_loss:.2f} | Train acc: {train_acc:.2f}' \
              f' | Val acc: {val_acc:.2f} | Val_loss: {val_loss} | lr: {epoch_lr} | Runtime: {epoch_time:.2f} seconds')
    
    return total_loss, epoch_times, accuracies_train, accuracies_validation
    
