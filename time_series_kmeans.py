import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, adjusted_rand_score
from scipy.signal import butter, filtfilt

# Cargar datos y transponer
file_path = "/Users/melaniealvarez/Documents/Investigacion/bonn_dataset_columns.csv"
data = np.loadtxt(file_path, delimiter=",").T

# Convertir a DataFrame
dataDF = pd.DataFrame(data)

# Separar etiquetas y datos
X = to_time_series_dataset(data[:, 1:])  # Las características
y = data[:, 0]  # Las etiquetas

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eliminar última dimensión extra
X_train = X_train.squeeze(-1)
X_test = X_test.squeeze(-1)

# Definir función de filtro pasa bajos
def low_pass_filter(signal, cutoff=40.0, fs=173.61, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Aplicar filtro a las señales EEG
cutoff_freq = 40.0  # Frecuencia de corte
fs = 173.61  # Frecuencia de muestreo
X_train_filtered = np.array([low_pass_filter(signal, cutoff=cutoff_freq, fs=fs) for signal in X_train])
X_test_filtered = np.array([low_pass_filter(signal, cutoff=cutoff_freq, fs=fs) for signal in X_test])

# Normalizar las series de tiempo
X_train_filtered = TimeSeriesScalerMeanVariance().fit_transform(X_train_filtered)

# Definir número de clusters
n_clusters = 2

# Aplicar TimeSeriesKMeans con DTW
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=42)
clusters = model.fit_predict(X_train_filtered)

# Evaluar el clustering con métricas supervisadas
adjusted_rand = adjusted_rand_score(y_train, clusters)
print(f"Adjusted Rand Index: {adjusted_rand:.4f}")

# Asignar clusters a etiquetas reales (requiere mapeo)
def map_clusters_to_labels(clusters, true_labels):
    from scipy.stats import mode
    true_labels = np.array(true_labels, dtype=int)  # Convertir etiquetas a enteros
    label_map = {}
    for cluster in np.unique(clusters):
        mask = (clusters == cluster)
        true_mode = mode(true_labels[mask], keepdims=True).mode[0]
        label_map[cluster] = true_mode
    return np.array([label_map[c] for c in clusters])

mapped_clusters = map_clusters_to_labels(clusters, y_train)

# Calcular métricas supervisadas
accuracy = accuracy_score(y_train, mapped_clusters)
recall = recall_score(y_train, mapped_clusters, average='weighted')
f1 = f1_score(y_train, mapped_clusters, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualizar los clusters
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    plt.subplot(1, n_clusters, cluster + 1)
    for series in X_train_filtered[clusters == cluster]:
        plt.plot(series.ravel(), "k-", alpha=0.3)
    plt.plot(model.cluster_centers_[cluster].ravel(), "r-", linewidth=2)
    plt.title(f"Cluster {cluster}")
plt.tight_layout()
plt.show()
