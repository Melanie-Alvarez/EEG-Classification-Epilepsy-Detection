# Proyecto de Agrupamiento Temporal para Señales EEG

Este repositorio contiene una serie de notebooks utilizados para evaluar distintos métodos de **clustering temporal** sobre señales EEG de los datasets **CHB-MIT** y **Bonn**, aplicando enfoques basados en **Time Series K-means (TSK-means)**, y **K-Shape**, junto con pruebas de modelado ARIMA para comparar comportamiento dinámico entre sujetos.

---

## 🧠 Datasets Utilizados

### 1. **CHB-MIT Scalp EEG Database**
- Fuente: PhysioNet  
- Sujetos: 23 pacientes pediátricos con epilepsia.
- Señales EEG multicanal (23 canales) segmentadas por crisis.
- Frecuencia de muestreo: 256 Hz.
- Usado para pruebas **Leave-One-Subject-Out (LOSO)** con métodos TSK-means y K-Shape.

### 2. **Bonn University Dataset**
- Fuente: Departamento de Epileptología, Universidad de Bonn.  
- Señales EEG de 5 conjuntos (A–E): normales, interictales y ictales.
- Frecuencia de muestreo: 173.61 Hz.

---

## ⚙️ Transformaciones Preliminares

### Notebook: `transformaciones.ipynb`
Incluye el preprocesamiento de señales:
- Normalización MinMax y z-score.
- Ventaneo temporal para dividir las señales en segmentos fijos.
- Transformaciones para suavizar ruido (filtros y moving average).
- Reducción de dimensionalidad con PCA cuando es necesario.

### Notebook: `generarDatos_CHB.ipynb`
Genera subconjuntos del CHB-MIT filtrando sujetos, canales y períodos específicos.  
Permite replicar las pruebas de clustering bajo distintas configuraciones de sujetos y ventanas.

---

## 🔬 Pruebas Realizadas

### 1. **K-Means con ARIMA (`Kmeans_Arima.ipynb`)**
- Clustering sobre coeficientes ARIMA ajustados a cada segmento de EEG.
- Objetivo: evaluar si las características temporales ajustadas con ARIMA pueden separarse en grupos homogéneos.
- Métricas: Inertia, Silhouette Score y visualización de centroides.

### 2. **Time Series K-Means (TSK-means)**
- Archivos: `TSKmeans_Bonn.ipynb`, `TSKmeans_CHB.ipynb`, `CHB_MIT_LOSO_TSKmeans.ipynb`
- Usa distancia DTW (Dynamic Time Warping) para medir similitud temporal.
- Pruebas con:
  - Validación LOSO (Leave-One-Subject-Out) en CHB-MIT.
  - Agrupamiento global en Bonn.
- Resultados:
  - Silhouette promedio ≈ 0.48–0.62 según dataset.
  - Grupos coherentes con fases ictales e interictales.

### 3. **K-Shape Clustering**
- Archivos: `Bonn_Kshape.ipynb`, `CHB_MIT_LOSO_Kshape.ipynb`
- Implementa alineación basada en cross-correlation.
- Centroides representativos de forma de onda promedio.
- Resultados:
  - Mejor separación entre estados normales y patológicos.
  - Permite visualización clara de patrones de forma de onda.

---

## 🧩 Validación LOSO

Los notebooks **CHB_MIT_LOSO_Kshape.ipynb** y **CHB_MIT_LOSO_TSKmeans.ipynb** emplean un esquema **Leave-One-Subject-Out**, donde cada paciente se usa como conjunto de prueba mientras el resto sirve para entrenamiento.  
Este enfoque permite:
- Evaluar la **generalización entre sujetos**.
- Evitar sobreajuste a individuos específicos.
- Analizar consistencia de patrones epilépticos comunes.

---

## 📈 Métricas y Visualizaciones

- **Silhouette Score**: calidad del agrupamiento.  
- **Davies–Bouldin Index**: separación entre clusters.  
- **Centroid plots**: forma promedio de las señales por grupo.  
- **Confusion maps**: proporción de segmentos clasificados por grupo.  
- **Visualización temporal**: reconstrucción de las señales agrupadas.

---

## 🧪 Conclusiones Generales

- **TSK-means** con DTW logró mayor estabilidad intersujeto.  
- **K-Shape** presentó centroides más interpretables en señales EEG.  
- El uso de **validación LOSO** confirmó la variabilidad entre sujetos, pero permitió identificar patrones epilépticos robustos.  
- Las transformaciones y ventanas de tiempo influyen significativamente en la calidad de agrupamiento.

---

## 💻 Requisitos de Ejecución

```bash
pip install tslearn numpy pandas matplotlib scipy scikit-learn statsmodels
