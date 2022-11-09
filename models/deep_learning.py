from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

class DeepNeuralNetwork:
    def __init__(self, learning_rate = 0.1, n_iter = 100, hidden_layers=[16, 16, 16]) -> None:
        self.learning_rate = learning_rate 
        self.n_iter = n_iter 
        self.hidden_layers = hidden_layers 
        
    def __repr__(self) -> str:
        return "DeepNeuralNetwork"
        
    def __init_params(self, dimensions):
        params = {}
        for dim in range(1, len(dimensions)):
            params['W' + str(dim)] = np.random.randn(dimensions[dim], dimensions[dim - 1])
            params['b' + str(dim)] = np.random.randn(dimensions[dim], 1)
        self.__params = params

    def __forward_propagation(self, X):
        activations = { "A0" : X }
        for dim in range(1, len(self.__params) // 2 + 1):
            Z = self.__params['W' + str(dim)].dot(activations['A' + str(dim - 1)]) + self.__params['b' + str(dim)]
            activations['A' + str(dim)] = 1 / (1 + np.exp(-Z))
        return activations

    def __back_propagation(self, y, activations):
        m = y.shape[1]
        C = len(self.__params) // 2
        dZ = activations['A' + str(C)] - y
        gradients = {}
        for dim in reversed(range(1, C + 1)):
            gradients['dW' + str(dim)] = 1/m * np.dot(dZ, activations['A' + str(dim - 1)].T)
            gradients['db' + str(dim)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if dim > 1:
                dZ = np.dot(self.__params['W' + str(dim)].T, dZ) * activations['A' + str(dim - 1)] * (1 - activations['A' + str(dim - 1)])
        return gradients

    def __update_params(self, gradients):
        for dim in range(1, len(self.__params) // 2 + 1):
            self.__params['W' + str(dim)] -= (self.learning_rate * gradients['dW' + str(dim)])
            self.__params['b' + str(dim)] -= (self.learning_rate * gradients['db' + str(dim)])

    def predict_proba(self, X):
        return self.__forward_propagation(X)['A' + str(len(self.__params) // 2)]
    
    def predict(self, X):
        return self.predict_proba(X) >= 0.5
    
    def log_loss(self, X, y):
        epsilon = 1e-15
        A = self.__forward_propagation(X)
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def train(self, X, y):
        dimensions = list(self.hidden_layers)
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])
        
        self.__init_params(dimensions)
        
        self.__train_loss = []
        self.__accuracy = []
        
        # Gradient Descent
        for i in tqdm(range(self.n_iter)):
            activations = self.__forward_propagation(X)
            gradients = self.__back_propagation(y, activations)
            self.__update_params(gradients)
            
            # Calculation of Log Loss and Accuracy
            if i%5 == 0:
                A = activations['A' + str(len(self.__params)//2)]
                self.__train_loss.append(log_loss(y, activations['A' + str(len(self.__params)//2)]))
                self.__accuracy.append(accuracy_score(y.flatten(), self.predict(X).flatten()))
            
    def train_loss(self):
        return self.__train_loss

    def accuracy(self):
        return self.__accuracy
    
    def training_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.__train_loss, label='Train Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.__accuracy, label='Train Accuracy')
        plt.legend()
        plt.show(block=False)

