import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import pandas as pd
import numpy as np  # Importar numpy
from sklearn.metrics import ConfusionMatrixDisplay  # Import ConfusionMatrixDisplay

def plot_roc_curve(y_true, y_probs, output_dir, phase):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {phase}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{phase}.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir, phase):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {phase}')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{phase}.png"))
    plt.close()


def save_classification_report(y_true, y_pred, output_dir, phase):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(output_dir, f"classification_report_{phase}.csv"), index=True)


def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization

    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

    return loss_values

def evaluate_model(model, data_loader, device, output_dir, phase):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {phase} set: {accuracy:.2f}%')

    y_true_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)
    y_probs_np = np.array(all_probs)


    plot_confusion_matrix(y_true_np, y_pred_np, output_dir, phase)
    save_classification_report(y_true_np, y_pred_np, output_dir, phase)
    
    return accuracy

def plot_loss(loss_values, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Pérdida', color='blue')
    plt.title('Función de Pérdida Durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()