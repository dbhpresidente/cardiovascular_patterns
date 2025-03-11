import os
import time
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from torchviz import make_dot
from torchview import draw_graph
import hiddenlayer as hl
import itertools

pd.set_option('display.max_columns', 500)

def main():
    # Verificar disponibilidad de GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Ruta al archivo CSV de anotaciones (asegúrate de ajustar la ruta)
    data_dir = '../../Maestría tecnología avanzada/project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'  # Directorio donde están los archivos .dat
    annotations_file = os.path.join(data_dir, 'df_info_ptbdb_xl.csv')

    # Cargar el archivo CSV con etiquetas y metadatos
    df = pd.read_csv(annotations_file)

    def read_ecg(file_name):
        sigs, fields = wfdb.rdsamp(file_name, channels=[i for i in range(12)], sampfrom=0)
        return pd.DataFrame(sigs, columns=fields['sig_name'])

    # Función para cargar datos de ECG en formato .dat
    def load_ecg_data(df, data_dir):
        signals = []
        labels = []

        for idx, row in df.iterrows():
            record = wfdb.rdsamp(data_dir + row['filename_hr'].replace('.mat', ''))
            ecg_signal = record[0]
            label = row['NOT_NORM']
            if ecg_signal.shape[0] >= 1000:
                ecg_signal = ecg_signal[:1000, :]
            else:
                ecg_signal = np.pad(ecg_signal, ((0, 1000 - ecg_signal.shape[0]), (0, 0)), 'constant')
            signals.append(ecg_signal)
            labels.append(label)

        signals = np.array(signals)
        labels = np.array(labels)
        return signals, labels

    # Cargar los datos y etiquetas
    signals, labels = load_ecg_data(df, data_dir)

    # Crear un mapeo de etiquetas a números
    label_mapping = {label: idx for idx, label in enumerate(df['diagnostic_superclass'].unique())}

    X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.3, random_state=123)

    # Convertir a tensores de PyTorch con la forma (n_samples, n_channels, seq_length)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Confirmar que la forma es correcta
    print("Shape of X_train after permutation:", X_train.shape)
    print("Shape of X_test after permutation:", X_test.shape)
    print("Shape of y_train:", y_train.shape)

    class ECGCNN(nn.Module):
        def __init__(self, num_classes):
            super(ECGCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
            self.fc1 = nn.Linear(64 * 247, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.contiguous().view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = torch.softmax(x, dim=1)
            return x

    # Crear TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Crear DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Definir hiperparámetros
    num_classes = len(set(y_train.cpu().numpy()))
    num_epochs = 50
    learning_rate = 0.001

    # Inicializar el modelo, el optimizador y la función de pérdida
    model = ECGCNN(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_values = []
    # Entrenar el modelo
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)  # Asegurarse de que ambos estén en el mismo dispositivo
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Graficar la función de pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Pérdida', color='blue')
    plt.title('Función de Pérdida Durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()
    plt.show()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Asegurarse de que ambos estén en el mismo dispositivo
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

    test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Asegurarse de que ambos estén en el mismo dispositivo
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

    y_pred = outputs
    y_probs = torch.softmax(y_pred, dim=1)[:, 1]
    y_pred_labels = torch.argmax(y_pred, dim=1)

    y_true_np = labels.cpu().numpy()
    y_pred_np = y_pred_labels.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    print("Reporte de Clasificación:")
    print(classification_report(y_true_np, y_pred_np))

    roc_auc = roc_auc_score(y_true_np, y_probs_np)
    print(f"ROC AUC: {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()

    conf_matrix = confusion_matrix(y_true_np, y_pred_np)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()

    classes = ["Clase 0", "Clase 1"]
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, f"{conf_matrix[i, j]}", 
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.show()

    input_tensor = torch.randn(1, 12, 1000).to(device)

    make_dot(model(input_tensor), params=dict(model.named_parameters())).render("ecgcnn_architecture", format="png")

    graph = draw_graph(model, input_tensor, expand_nested=True)
    graph.visual_graph.render("ecg_model_architecture", format="png", cleanup=True)

if __name__ == "__main__":
    main()

