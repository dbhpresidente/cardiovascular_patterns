import os
import wfdb
import numpy as np
import pandas as pd

def read_ecg(file_name):
    """
    Lee un archivo de ECG y devuelve las señales y los campos en un DataFrame.

    Args:
        file_name (str): Ruta al archivo de ECG.

    Returns:
        pd.DataFrame: DataFrame con las señales de ECG.
    """
    sigs, fields = wfdb.rdsamp(file_name, channels=[i for i in range(12)], sampfrom=0)
    return pd.DataFrame(sigs, columns=fields['sig_name'])

def load_ecg_data(df, data_dir):
    """
    Carga los datos de ECG y las etiquetas desde un DataFrame y un directorio de datos.

    Args:
        df (pd.DataFrame): DataFrame con las anotaciones y metadatos.
        data_dir (str): Directorio donde se encuentran los archivos de datos de ECG.

    Returns:
        tuple: Una tupla con dos elementos:
            - signals (np.array): Array con las señales de ECG.
            - labels (np.array): Array con las etiquetas correspondientes.
    """
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