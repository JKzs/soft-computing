import numpy as np

class Perceptron():
    """
    Implementación básica de un perceptrón para clasificación binaria

    Parámetros:
        eta: float
            Tasa de aprendizaje (generalmente entre 0.0 y 1.0)
        epochs: int
            Número de iteraciones sobre el dataset de entrenamiento

    Atributos:
        weights: ndarray
            Pesos del perceptrón después del entrenamiento
        bias: float
            Término independiente (sesgo)
        errors_: list
            Lista con el número de errores de clasificación por cada época
    """


    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs
    
    def train(self, X, y):
        """
        Entrena el perceptrón con los datos de entrada
        
        Parámetros:
            X: ndarray, shape = [n_muestras, n_features]
                Datos de entrenamiento.
            y: ndarray, shape = [n_muestras]
                Etiquetas de clase (0 o 1)
        
        Retorna:
            self: objeto
                Perceptrón entrenado
        """
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                update = self.eta * (yi - y_pred)
                
                self.weights += update * xi
                self.bias += update
                
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        return self 
    
    def net_input(self, X):
        """
        Calcula la combinación lineal (z = w·x + b)

        Parámetros:
            X: ndarray
                Vector de características.
        
        Retorna:
            float
                Valor neto de entrada (antes de la función de activación)
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Predice la clase (0 o 1) para una entrada X.

        Parámetros:
            X: ndarray
                Vector de características
        
        Retorna:
            int o ndarray
                Clase predicha (0 o 1)
        """
        z = self.net_input(X)
        return np.where(z >= 0.0, 1, 0)
