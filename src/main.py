import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_ecg_data
from model import ECGCNN
from train import train_model, evaluate_model, plot_loss, hyperparameter_tuning
from visualize import plot_roc_curve, plot_confusion_matrix, visualize_model

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

    # Dividir los datos de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

    # Crear TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Crear DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Definir hiperparámetros
    num_classes = len(set(y_train.cpu().numpy()))
    num_epochs = 100  # Incrementar el número de épocas
    learning_rate = 0.0001  # Reducir la tasa de aprendizaje

    # Inicializar el modelo
    model = ECGCNN(num_classes)
    model = model.to(device)

    # Crear directorio de salida para las imágenes
    output_img_dir = "img"
    os.makedirs(output_img_dir, exist_ok=True)

    # Crear directorio de salida para los archivos CSV
    output_doc_dir = "doc"
    os.makedirs(output_doc_dir, exist_ok=True)

    # Ajuste de hiperparámetros
    best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, output_doc_dir, output_img_dir)
    print("Mejores hiperparámetros:", best_params)

    # Entrenar el modelo con los mejores hiperparámetros
    model = ECGCNN(num_classes, dropout_rate=best_params['dropout_rate'])
    model = model.to(device)
    loss_values = train_model(model, train_loader, best_params['num_epochs'], best_params['learning_rate'], device)

    # Graficar la función de pérdida
    plot_loss(loss_values, output_img_dir)

    # Evaluar el modelo en el conjunto de entrenamiento
    evaluate_model(model, train_loader, device, output_img_dir, "train", output_doc_dir)

    # Evaluar el modelo en el conjunto de prueba
    evaluate_model(model, test_loader, device, output_img_dir, "test", output_doc_dir)

    # Visualizar el modelo
    input_tensor = torch.randn(1, 12, 1000).to(device)
    visualize_model(model, input_tensor, device, output_img_dir)

if __name__ == "__main__":
    main()