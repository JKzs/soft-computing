import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from perceptrones.simple import Perceptron

"""
Ejemplo de uso de un Perceptrón para implementar una compuerta lógica OR.

1. Define los datos de entrada (tabla de verdad de la compuerta OR).
2. Entrena un perceptrón simple con tasa de aprendizaje eta=0.1.
3. Visualiza la frontera de decisión usando mlextend.

Requisitos:
- numpy
- matplotlib
- mlextend
- perpectrones.simple (implementación propia)
"""
#Datos de entrenamiento: Tabla de verdad de OR
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0,
              1,
              1,
              1])

#Inicialización del perceptrón 
#eta: tasa de aprendizaje
#epochs: número de iteraciones sobre el dataset
per = Perceptron(eta=0.1, epochs=10)
per.train(X, y)

#Visualización de la frontera de decisión
plot_decision_regions(X, y, clf=per, legend=2)
plt.xlabel("x1 (entrada 1)")
plt.ylabel("x2 (entrada 2)")
plt.title("Frontera de decisión - Perceptrón (AND)")
plt.show()

