import os
import numpy as np
import scipy.io
import random

def cargar_datos_mat(archivo_mat, claves):
    """
    Carga los datos desde un archivo .mat bajo varias claves y retorna un array NumPy
    combinando los datos de todas las claves.
    """
    datos = scipy.io.loadmat(archivo_mat)
    datos_totales = []

    # Iterar sobre las claves especificadas
    for clave in claves:
        if clave in datos:
            print(f"Cargando datos de la clave '{clave}' del archivo {archivo_mat}")
            
            # Verificar si el valor bajo la clave es de tipo numpy.ndarray
            if isinstance(datos[clave], np.ndarray):
                datos_totales.append(datos[clave])
            else:
                raise ValueError(f"Los datos bajo '{clave}' en {archivo_mat} no son un array NumPy.")
        else:
            print(f"La clave '{clave}' no existe en el archivo {archivo_mat}.")
    
    # Combinar los datos de todas las claves horizontalmente
    if len(datos_totales) > 0:
        return np.hstack(datos_totales)
    else:
        raise ValueError(f"No se encontraron datos válidos en las claves {claves} en el archivo {archivo_mat}.")

def guardar_datos_txt(archivos_mat, archivo_salida, claves):
    """
    Carga varios archivos .mat bajo las claves especificadas y guarda sus datos
    en formato .txt donde cada columna es un archivo diferente, y la primera fila
    contiene un identificador para cada columna.
    """
    # Lista para almacenar los datos de todos los archivos .mat
    datos_todos = []
    identificadores = []

    for archivo in archivos_mat:
        print(f"Cargando {archivo}...")
        
        # Identificar el tipo de archivo (interictal, preictal, ictal) por el nombre del directorio
        if 'interictal' in archivo.lower():  # Verificar interictal primero para evitar conflicto con 'ictal'
            identificador = 0
        elif 'preictal' in archivo.lower():
            identificador = 1
        elif 'ictal' in archivo.lower():
            identificador = 2
        else:
            raise ValueError(f"No se puede identificar el tipo de archivo: {archivo}")
        
        # Cargar los datos del archivo .mat
        datos = cargar_datos_mat(archivo, claves)
        
        # Asegúrate de que los datos estén en formato de columna (transpuesta si es necesario)
        if datos.ndim == 1:  # Si es un vector 1D
            datos = datos[:, np.newaxis]  # Convertir a una matriz columna
        elif datos.shape[0] < datos.shape[1]:
            datos = datos.T  # Transponer si hay más columnas que filas
        
        # Añadir los datos a la lista
        datos_todos.append(datos)
        
        # Añadir el identificador correspondiente
        identificadores.append(identificador)
    
    # Combinar todos los datos en una sola matriz de NumPy, concatenando horizontalmente (columnas)
    datos_comb = np.hstack(datos_todos)

    # Crear la primera fila de identificadores
    fila_identificadores = np.array(identificadores)[np.newaxis, :]
    
    # Añadir la fila de identificadores antes de los datos
    datos_final = np.vstack([fila_identificadores, datos_comb])

    # Guardar los datos combinados en un archivo .txt
    np.savetxt(archivo_salida, datos_final, delimiter=',')

    print(f"Datos guardados en {archivo_salida}")

# Configurar directorios de los archivos .mat
directorio1 = 'EEG Epilepsy Datasets/ictal'  # Cambia a la ruta del primer conjunto de archivos .mat
directorio2 = 'EEG Epilepsy Datasets/interictal'  # Cambia a la ruta del segundo conjunto de archivos .mat
directorio3 = 'EEG Epilepsy Datasets/preictal'  # Cambia a la ruta del tercer conjunto de archivos .mat

# Obtener todos los archivos .mat de los tres directorios
archivos_mat = [os.path.join(directorio1, f) for f in os.listdir(directorio1) if f.endswith('.mat')]
archivos_mat += [os.path.join(directorio2, f) for f in os.listdir(directorio2) if f.endswith('.mat')]
archivos_mat += [os.path.join(directorio3, f) for f in os.listdir(directorio3) if f.endswith('.mat')]

# Verificar si hay al menos 150 archivos .mat
if len(archivos_mat) < 150:
    raise ValueError("No hay suficientes archivos .mat. Se necesitan al menos 150 archivos.")

# Dividir aleatoriamente el 80% para entrenamiento y 20% para prueba
random.shuffle(archivos_mat)
n_train = int(0.8 * len(archivos_mat))

archivos_train = archivos_mat[:n_train]
archivos_test = archivos_mat[n_train:]

# Definir las claves que contienen los datos en los archivos .mat
claves_datos = ['ictal', 'preictal', 'interictal']  # Ajusta los nombres de las claves según tu archivo

# Guardar los datos de entrenamiento y prueba en archivos separados
guardar_datos_txt(archivos_train, 'ECGKhas_Train.txt', claves_datos)
guardar_datos_txt(archivos_test, 'ECGKhas_Test.txt', claves_datos)
