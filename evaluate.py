import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def evaluate(model, data_loader, device, class_names):
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
    
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.jpg')
    plt.close()
    
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'misclassified_indices': misclassified_indices
    }