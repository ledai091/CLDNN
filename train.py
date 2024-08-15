import time
from datetime import timedelta
import torch
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device):
    model.to(device)
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(model.device)
            y = y.to(model.device)
            
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(model.device)
                y = y.to(model.device)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        epochs_left = epochs - (epoch + 1)
        eta_seconds = epochs_left * epoch_duration
        eta = str(timedelta(seconds=int(eta_seconds)))
        
        print(f'Epoch: {epoch + 1:3d}/{epochs:<3d} | '
            f'Train Loss: {train_losses[-1]:<10.4f} | '
            f'Val Loss: {val_losses[-1]:<10.4f} | '
            f'Epoch Time: {epoch_duration:<7.2f}s | '
            f'ETA: {eta:<8}')
    
    total_time = time.time() - start_time
    print(f'Total training time: {str(timedelta(seconds=int(total_time)))}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('loss_chart.png')
    print("Loss chart saved as 'loss_chart.png'")
    
    return train_losses, val_losses