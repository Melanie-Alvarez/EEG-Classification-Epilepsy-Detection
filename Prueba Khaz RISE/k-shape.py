import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support

# Cargar los datasets, ignorando la primera columna (que contiene las etiquetas)
data_train = np.loadtxt("ECGKhas_Train.txt", delimiter="\t").T  # Transponer los datos para tener las series temporales en filas
X_train = to_time_series_dataset(data_train[:, 1:])
y_true = data_train[:, 0].astype(int)

data_test = np.loadtxt("ECGKhas_Test.txt", delimiter="\t").T  # Omitir la primera columna para los datos de prueba
X_test = to_time_series_dataset(data_test[:, 1:])  # Las series temporales están ahora en filas
y_true_test = data_test[:, 0].astype(int)  # Etiquetas verdaderas en la primera fila del conjunto de prueba

# Escalar las series temporales para la media y la varianza
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Estandarizar para tener media 0 y varianza 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Estadísticas básicas
print("Número de series temporales en el conjunto de entrenamiento:", X_train.shape[0])
print("Número de series temporales en el conjunto de prueba:", X_test.shape[0])
print("Longitud de las series temporales:", X_train.shape[1])

# Inicializar el modelo K-Shape
n_clusters = 3  # Ajusta este valor dependiendo de tu problema
ks = KShape(n_clusters=n_clusters, n_init=1, random_state=0)

# Ajustar el modelo
ks.fit(X_train_scaled)

# Predecir los clusters para el conjunto de entrenamiento
y_pred_train = ks.predict(X_train_scaled)
print("Predicciones (entrenamiento):", y_pred_train)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_train)
print("Matriz de confusión (Train):")
print(conf_matrix)

# Reporte de clasificación (Precision, Recall, F1-Score)
report = classification_report(y_true, y_pred_train)
print("Reporte de clasificación (Train):")
print(report)

# Calcular precisión, recall, F1 y soporte para cada clase
precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred_train, average='macro')
print(f"Precision (macro): {precision}")
print(f"Recall (macro): {recall}")
print(f"F1-Score (macro): {fscore}")

# Visualizar los centroides de los clusters y las series originales
for yi in range(n_clusters):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Cluster {yi + 1} Series Originales (entrenamiento)")
    for xx in X_train_scaled[y_pred_train == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Cluster {yi + 1} Centroide (entrenamiento)")
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.grid(True)
    
    plt.show()

