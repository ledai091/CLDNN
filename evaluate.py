import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_and_save_confusion_matrix(cm, classes, file_name='confusion_matrix.jpg'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
    

def evaluate(model, data_loader, device, classes):
    model.eval()
    all_preds = []
    all_targets = []
    misclassified_indices = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            start_idx = batch_idx * data_loader.batch_size
            batch_misclassified = (predicted != y).nonzero(as_tuple=True)[0]
            misclassified_indices.extend(start_idx + batch_misclassified.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    
    plot_and_save_confusion_matrix(cm, classes, file_name="confusion_matrix.png")
    
    return acc, cm, misclassified_indices