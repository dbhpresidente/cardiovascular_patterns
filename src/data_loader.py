import os
import wfdb
import numpy as np
import pandas as pd

def read_ecg(file_name):
    sigs, fields = wfdb.rdsamp(file_name, channels=[i for i in range(12)], sampfrom=0)
    return pd.DataFrame(sigs, columns=fields['sig_name'])

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