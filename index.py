from matplotlib import pyplot as plt
import numpy as np
from datasets import load_catdog

X_train, y_train, X_test, y_test = load_catdog()

X_train_reshape = X_train.reshape(X_train.shape[0], -1).T / X_train.max()
y_train = y_train.T

X_test_reshape = X_test.reshape(X_test.shape[0], -1).T / X_train.max()
y_test = y_test.T

from models.deep_learning import DeepNeuralNetwork

model = DeepNeuralNetwork(learning_rate = 0.01, n_iter = 5000, hidden_layers=[32, 32, 32])

print('Model training loading...')
model.train(X_train_reshape, y_train)
print('Model training completed.')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

while True:
    file_path = input('\nLoad an image of cat or dog, the image size should be 64x64.\n')
    try:
        image = rgb2gray(plt.imread(file_path))
        image = image.ravel().reshape(image.ravel().shape[0], 1)
        is_dog = model.predict(image)
        is_dog_text = "dog" if is_dog else "cat"
        print(f"This animal is a {is_dog_text}.")
    except Exception as E:
        print(E)
    finally:
        print('Would you repeat or Exit?\n')
        print('1 - Repeat\n')
        print('2 - Exit\n')
        choice = input()
        if choice != '1':
            break