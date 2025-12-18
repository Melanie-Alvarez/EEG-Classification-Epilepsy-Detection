import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.clustering import KShape
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

data = np.loadtxt("/Users/melaniealvarez/Documents/Investigacion/bonn_dataset_columns_correct.csv", delimiter=",")[:, :].T
dataDF = pd.DataFrame(data)
X = to_time_series_dataset(data[:, 1:])
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.squeeze(-1)  # Elimina la última dimensión
X_test = X_test.squeeze(-1)

def low_pass_filter(signal, cutoff=40.0, fs=173.61, order=5):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Normalización de la frecuencia de corte
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)  # Aplicar filtro

# Parámetros del filtro
cutoff_freq = 40.0  # Frecuencia de corte de 40 Hz
fs = 173.61  # Frecuencia de muestreo

# 📌 Filtrar señales EEG
X_train_filtered = np.array([low_pass_filter(signal, cutoff=cutoff_freq, fs=fs) for signal in X_train])
X_test_filtered = np.array([low_pass_filter(signal, cutoff=cutoff_freq, fs=fs) for signal in X_test])



# Lista de escaladores a probar
scalers = {
    'TimeSeriesScalerMeanVariance': TimeSeriesScalerMeanVariance(mu=0., std=1.),
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler()
}

# Parámetros para GridSearch
param_grid = {
    'n_clusters': [2],  # Número de clusters
    'n_init': [3, 5, 7],      # Inicializaciones
    'tol': [1e-3, 1e-5, 1e-6]       # Tolerancia
}

# Almacenar resultados globales
best_overall_score = -np.inf
best_overall_params = None
best_overall_model = None
best_scaler_name = None

# Probar cada escalador
for scaler_name, scaler in scalers.items():
    print(f"\nProbando con escalador: {scaler_name}")

    # Escalar los datos (ajustando la forma para sklearn si es necesario)
    if scaler_name == 'TimeSeriesScalerMeanVariance':
        X_train_scaled = scaler.fit_transform(X_train_filtered)
        X_test_scaled = scaler.transform(X_test_filtered)
    else:
        # Aplicar escalado
        X_train_scaled = scaler.fit_transform(X_train_filtered)
        X_test_scaled = scaler.transform(X_test_filtered)

    # Evaluación manual de los parámetros
    best_score = -np.inf
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"Probando configuración: {params}")
        model = KShape(n_clusters=params['n_clusters'], n_init=params['n_init'], tol=params['tol'], random_state=0)
        model.fit(X_train_scaled)

        # Predicciones
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Evaluación supervisada
        acc = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_train, y_pred_train, average='weighted')
        recall = recall_score(y_train, y_pred_train, average='weighted')
        f1 = f1_score(y_train, y_pred_train, average='weighted')

        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

        if acc > best_score:  # Se puede cambiar a otra métrica si se prefiere
            best_score = acc
            best_params = params
            best_model = model

    print(f"Mejor configuración para {scaler_name}: {best_params}")
    print(f"Mejor Accuracy para {scaler_name}: {best_score:.4f}")

    # Actualizar mejor global si es necesario
    if best_score > best_overall_score:
        best_overall_score = best_score
        best_overall_params = best_params
        best_overall_model = best_model
        best_scaler_name = scaler_name
        best_X_test_scaled = X_test_scaled

# Imprimir la mejor configuración global encontrada
print("\n====== Mejor Configuración Global ======")
print(f"Mejor escalador: {best_scaler_name}")
print("Mejor configuración:", best_overall_params)
print("Mejor Accuracy:", best_overall_score)

# Evaluación final con el mejor modelo en el conjunto de prueba
y_pred_test = best_overall_model.predict(best_X_test_scaled)

# Matriz de confusión y reporte de clasificación
conf_matrix = confusion_matrix(y_train, y_pred_train)
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nReporte de clasificación:")
print(classification_report(y_train, y_pred_train))