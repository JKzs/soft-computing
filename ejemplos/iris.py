import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from perceptrones.simple import Perceptron

"""
Uso de un perceptrón simple para clasificación binaria de flores Iris.

1. Se carga el dataset Iris desde la UCI Machine Learning Repository.
2. Se seleccionan las 100 primeras muestras (Iris-setosa e Iris-versicolor)
   y solo dos características: longitud del sépalo (columna 0) y longitud
   del pétalo (columna 2).
3. Se codifican las clases en valores binarios:
   -Iris-setosa -> 0
   -Iris-versicolor -> 1
4. Se entrena un perceptrón simple con tasa de aprendizaje eta=0.1 y
   epochs=10
5. Se visualiza:
   -La frontera de decisión generada por el perceptrón.
   -La evolución del número de errores de clasificación por iteración.

Requisitos:
   -pandas
   -numpy
   -matplotlib
   -mlextend
   -perceptrones.simple (implementación propria del perceptrón)
"""
#1. Cargar el dataset Iris
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None)

#2. Seleccionar las primeras 100 muestras y las columnas 0 (sépalo) y 2 (pétalos)
X = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values

#3. Codificar las etiquetas: setosa=0, versicolor=1
y = np.where(y=='Iris-setosa', 0, 1)

#4. Inicializar y entrenar el perceptrón
per = Perceptron(eta=0.1, epochs=10)
per.train(X, y)

#5a. Visualización de la frontera de decisión
plot_decision_regions(X, y, clf=per)
plt.title('Perceptron - Clasificación de Iris')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

#5b. Curva de errores de clasificación.
plt.plot(range(1, len(per.errors_)+1), per.errors_, marker='o')
plt.title('Evolución del error durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Número de errores de clasificación')
plt.show()