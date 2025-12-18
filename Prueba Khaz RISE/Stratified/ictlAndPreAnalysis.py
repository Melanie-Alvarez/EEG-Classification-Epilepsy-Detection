import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support

# Cargar los datasets, incluyendo las etiquetas en la primera columna
data_train = np.loadtxt("ECGKhas_Train_Strata_Binary.txt", delimiter="\t")[:, :2].T  # Cargar todos los datos
X_train = to_time_series_dataset(data_train[:, 1:])  # Las series temporales están ahora en filas
y_true = data_train[:, 0].astype(int)  # Cargar las etiquetas verdaderas de la primera columna

data_test = np.loadtxt("ECGKhas_Test_Strata_Binary.txt", delimiter="\t")[:, :].T
X_test = to_time_series_dataset(data_test[:, 1:])  # Las series temporales están ahora en filas
y_test = data_test[:, 0].astype(int)  # Cargar las etiquetas verdaderas de la primera columna

# Escalar las series temporales para la media y la varianza
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Estandarizar para tener media 0 y varianza 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Número de series temporales en el conjunto de entrenamiento:", X_train.shape[0])
print("Número de series temporales en el conjunto de prueba:", X_test.shape[0])
print("Longitud de las series temporales:", X_train.shape[1])

# Inicializar el modelo K-Shape
n_clusters = 2  # Ajusta este valor dependiendo de tu problema
ks = KShape(n_clusters=n_clusters, n_init=1, random_state=0)

# Ajustar el modelo con los datos de entrenamiento
ks.fit(X_train_scaled)

# Predecir los clusters para los datos de entrenamiento
"""y_pred_train = ks.predict(X_train_scaled)
print("Predicciones (Train):", y_pred_train)"""

# Predecir los clusters para los datos de prueba
y_pred_test = ks.predict(X_test_scaled)
print("Predicciones (Test):", y_pred_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Matriz de confusión (Train):")
print(conf_matrix)

# Reporte de clasificación (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred_test)
print("Reporte de clasificación (Test):")
print(report)

# Calcular precisión, recall, F1 y soporte para cada clase
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
print(f"Precision (macro): {precision}")
print(f"Recall (macro): {recall}")
print(f"F1-Score (macro): {fscore}")

# Visualización de los centroides de los clusters y las series originales (opcional)
for yi in range(n_clusters):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Cluster {yi + 1} Series Originales")
    for xx in X_test[y_pred_test == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Cluster {yi + 1} Series Escaladas con Centroide")
    for xx in X_test_scaled[y_pred_test == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.grid(True)
    
    plt.show()