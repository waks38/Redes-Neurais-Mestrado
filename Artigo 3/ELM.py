import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class ELM:
    def __init__(self, n_hidden, random_state=0):
        """
        n_hidden: int (número de neurônios na camada oculta)
        """
        self.n_hidden = n_hidden
        self.random_state = random_state
        
        self.weights_input = None   # W (n_features, n_hidden)
        self.bias_hidden = None     # b (n_hidden,) -> Viés aleatório é crucial para ELM
        self.weights_output = None  # Beta (n_hidden + 1, 1)

    def _sigmoid(self, x):
        # Clip para evitar overflow numérico na exponencial
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y, lamb=0.00):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1. Inicialização Aleatória (Input Weights + Hidden Bias)
        self.weights_input = rng.uniform(-1., 1., (n_features, self.n_hidden))
        self.bias_hidden = rng.uniform(-1., 1., (self.n_hidden,))

        # 2. Projeção na Camada Oculta
        # H = sigmoid(XW + b)
        projection = X @ self.weights_input + self.bias_hidden
        H = self._sigmoid(projection)

        # 3. Adiciona bias para a camada de SAÍDA (para a regressão linear final)
        H_aug = np.column_stack([np.ones(n_samples), H])

        # 4. Ridge Regression (Mínimos Quadrados Regularizados)
        # Beta = (H^T H + λI)^-1 H^T y
        # CORREÇÃO: Sinal de + na regularização
        I = np.eye(H_aug.shape[1])
        
        # Opcional: Não regularizar o bias (primeiro elemento da diagonal = 0)
        I[0, 0] = 0 

        A = H_aug.T @ H_aug + lamb * I
        B = H_aug.T @ y.reshape(-1, 1)

        # Usando solve para estabilidade numérica (melhor que inv)
        self.weights_output = np.linalg.solve(A, B)
        
        return self

    def predict(self, X):
        n_samples = X.shape[0]

        # Projeção na camada oculta
        projection = X @ self.weights_input + self.bias_hidden
        H = self._sigmoid(projection)

        # Adiciona bias para a camada de saída
        H_aug = np.column_stack([np.ones(n_samples), H])

        # Predição final
        y_pred = H_aug @ self.weights_output
        
        return y_pred.ravel()
    
    def plotar_contornos(self, X, y, grid_size=100):
        # Define o grid para plotagem
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Prediz os valores no grid
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        Z = Z > 0  # Converte regressão da ELM em decisão binária (True/False)
        


        # Plota os contornos
        plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=ListedColormap(['#E0E0E0',"#7A7A7A","#363636"]), alpha=0.6)
        plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['#E0E0E0',"#363636"]))
        plt.title('Contornos da ELM')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()