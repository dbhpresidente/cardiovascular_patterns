import torch
import torch.nn as nn

class ECGCNN(nn.Module):
    """
    Clase que define la arquitectura de la red neuronal convolucional para la clasificación de señales ECG.

    Args:
        num_classes (int): Número de clases de salida.
    """
    def __init__(self, num_classes):
        """
        Inicializa la arquitectura de la red neuronal.

        Args:
            num_classes (int): Número de clases de salida.
        """
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 121, 256)  # Ajustar la dimensión aquí
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Define el paso hacia adelante de la red neuronal.

        Args:
            x (torch.Tensor): Tensor de entrada con forma (batch_size, n_channels, seq_length).

        Returns:
            torch.Tensor: Tensor de salida con las probabilidades de cada clase.
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x