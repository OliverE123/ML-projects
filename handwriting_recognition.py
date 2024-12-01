import numpy as np
import pandas as pd
import pickle
from asgiref.typing import WWWScope
from django.db.models import Index
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/oliverexter/Documents/Programming/Machine learning/mnist_test.csv')

print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)


data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0

def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(10, 64) * np.sqrt(2. / 64)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(0,Z)

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    log_probs = -np.log(A2[Y, np.arange(m)])
    loss = np.sum(log_probs) / m
    return loss

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    num_classes = 10
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLu(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1 , W2, b2, dW1, db1, dW2, db2, alpha):
     W1 = W1 - alpha * dW1
     b1 = b1 - alpha * db1
     W2 = W2 - alpha * dW2
     b2 = b2 - alpha * db2
     return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, batch_size=64):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]

    for i in range(iterations):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[j:j + batch_size]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            loss = compute_loss(A2, Y)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return W1, b1, W2, b2

W1, b1, W2 , b2 = gradient_descent(X_train, Y_train, 600, 0.01)


with open('model_params.pkl', 'wb') as f:
    pickle.dump((W1, b1, W2, b2), f)
print("Model saved successfully.")


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1,W2, b2):
    current_image = X_train [:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:" , prediction)
    print("Label:", label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


with open('model_params.pkl', 'rb') as f:
    W1, b1, W2, b2 = pickle.load(f)
print("Model loaded successfully.")

test_prediction(11, W1, b1, W2, b2)


