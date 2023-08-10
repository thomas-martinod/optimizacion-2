# Desarrollador: Thomas Martinod
# 1er Examen, Optimizacion 2

# Importaciones
import pandas as pd
import numpy as np

# Funcion para calcular la entropia de Shannon
def calcular_entropia_shannon(p):
    return -np.sum(p * np.log2(p))

# Funcion para calcular los pesos utilizando el metodo de Shannon
def calcular_pesos_shannon(matriz_datos):
    n, m = matriz_datos.shape
    pesos = []

    # Calcular los pesos de cada criterio usando el metodo de Shannon
    for j in range(m):
        p_j = matriz_datos[:, j] / np.sum(matriz_datos[:, j])
        entropia_j = calcular_entropia_shannon(p_j)
        pesos.append(1 - entropia_j / np.sum([1 - calcular_entropia_shannon(matriz_datos[:, k] / np.sum(matriz_datos[:, k])) for k in range(m)]))

    # Normalizar los pesos obtenidos
    suma_pesos = sum(pesos)
    pesos_normalizados = [w / suma_pesos for w in pesos]

    return pesos_normalizados

# Funcion para generar una matriz de comparacion aleatoria
def generar_matriz_comparacion_aleatoria(n):
    # Generar una matriz triangular superior con valores aleatorios (ejemplo: 1-9)
    matriz_superior = np.random.randint(1, 10, size=(n, n))
    # Hacer la matriz triangular inferior su valor reciproco
    matriz_inferior = 1 / matriz_superior.T
    # Combinar las matrices para obtener la matriz de comparacion
    matriz_comparacion = matriz_superior + matriz_inferior
    # Rellenar la diagonal con unos
    np.fill_diagonal(matriz_comparacion, 1)
    return matriz_comparacion

# Funcion para calcular los pesos utilizando el metodo AHP
def calcular_pesos_ahp(matriz_comparacion_criterios):
    m , n = matriz_comparacion_criterios.shape
    
    # Paso 2: Calcular los valores promedio de comparacion de criterios
    w_criterios = np.mean(matriz_comparacion_criterios, axis=0)
    
    # Paso 3: Normalizar los pesos de los criterios
    w_criterios /= np.sum(w_criterios)
    
    # Generar una matriz de comparacion aleatoria para las alternativas
    matriz_comparacion_alternativas = generar_matriz_comparacion_aleatoria(n)
    
    # Calcular los pesos para las alternativas
    w_alternativas = np.mean(matriz_comparacion_alternativas, axis=0)
    w_alternativas /= np.sum(w_alternativas)
    
    # Calcular la matriz ponderada
    matriz_ponderada = matriz_comparacion_alternativas * w_criterios[:, np.newaxis]
    
    # Calcular los pesos totales para cada alternativa
    w_totales = np.mean(matriz_ponderada, axis=1)
    w_totales /= np.sum(w_totales)  # Normalizar los pesos
    
    return w_totales

# Funcion principal
def main():
    # Leer los datos desde el archivo de texto y crear un DataFrame
    df = pd.read_csv("multimeters_data.txt", sep="\t")
    # Eliminar la columna "Multimetro"
    df = df.drop("Multimeter", axis=1)

    # Obtener el numero de filas m y columnas n en el DataFrame
    m, n = df.shape

    # Tipo de optimizacion para cada criterio
    # Crear un diccionario para especificar el tipo de optimizacion para cada columna
    typeOfOptimization = {
        col: "max" if (col != "Price" and col != "Weight") else "min"  # Establecer "Price" en "min", otros en "max"
        for col in df.columns
    }

    # Calcular las normas de las columnas utilizando NumPy
    norms = np.linalg.norm(df, axis=0)

    # Normalizar el DataFrame dividiendo cada columna por su norma
    df = df.divide(norms, axis=1)

    # Calcular los pesos utilizando el metodo AHP
    w_AHP = calcular_pesos_ahp(df.values)
    pesos_AHP_dict = {col: peso for col, peso in zip(df.columns, w_AHP)}

    # Calcular los pesos utilizando el metodo de Shannon
    w_shannon = calcular_pesos_shannon(df.values)
    pesos_shannon_dict = {col: peso for col, peso in zip(df.columns, w_shannon)}

    # Aplicar los pesos de Shannon y realizar el metodo TOPSIS
    FinalTOPSIS(df, pesos_AHP_dict, typeOfOptimization, m)

# Funcion para aplicar los pesos y realizar el metodo TOPSIS
def FinalTOPSIS(df, w, typeOfOptimization, m):
    # Aplicar los pesos de Shannon al DataFrame normalizado
    for col in df.columns:
        df[col] *= w[col]

    # Encontrar las alternativas ideales y anti-ideales para cada criterio
    idealA = {}
    antiidealA = {}

    for col in df.columns:
        max_value = df[col].max()
        min_value = df[col].min()

        if typeOfOptimization[col] == "max":
            idealA[col] = max_value
            antiidealA[col] = min_value

        elif typeOfOptimization[col] == "min":
            idealA[col] = min_value
            antiidealA[col] = max_value

    # Calcular las distancias ideales positivas (di_pos) y negativas (di_neg) para cada A_i
    di_pos = np.sqrt(np.sum((df - idealA)**2, axis=1))
    di_neg = np.sqrt(np.sum((df - antiidealA)**2, axis=1))

    # Calcular el Ratio de Similitud (RSi)
    RSi = [di_neg[i] / (di_pos[i] + di_neg[i]) for i in range(m)]

    # Crear una lista de tuplas con indices y valores RSi
    indexed_RSi = [(index + 1, value) for index, value in enumerate(RSi)]

    # Ordenar la lista de tuplas segun el valor de RSi en orden descendente
    sorted_RSi = sorted(indexed_RSi, key=lambda x: x[1], reverse=True)

    # Imprimir los valores ordenados de RSi y los indices correspondientes
    print("Best to worst multimeters:")
    for item in sorted_RSi:
        print(item)

# Llamar a la funcion principal si este script se ejecuta directamente
if __name__ == "__main__":
    main()
