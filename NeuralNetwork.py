import numpy as np


class NeuralNetwork:

    def __init__(self, layers, lr, epoch):
        self.x = None
        self.y = None
        self.layers = layers
        self.lr = lr
        self.epoch = epoch
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.loss = []

    def init_weights_and_bias(self):
        np.random.seed(4)

        self.params['w1'] = np.random.rand(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.rand(self.layers[1], )
        self.params['w2'] = np.random.rand(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.rand(self.layers[2], )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def dRelu(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    def entropy_loss(self, y, yhat):
        loss = -0.5 * (y - yhat) ** 2
        return loss

    def forward_propagation(self):
        z1 = self.x.dot(self.params['w1']) + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        yhat = self.sigmoid(z2)
        loss = self.entropy_loss(self.y, yhat)

        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['a2'] = yhat

        return yhat, loss

    def back_propagation(self, yhat):
        sigma_o = -(self.y - yhat) * yhat * (1 - yhat)
        dw2 = np.dot(self.cache['a1'].T, sigma_o)
        db2 = np.sum(dw2, axis=0, keepdims=True)

        sigma_h = sigma_o.dot(self.params['w2'].T) * self.dRelu(self.cache['z1'])
        dw1 = self.x.T.dot(sigma_h)
        db1 = np.sum(dw1, axis=0, keepdims=True)

        self.grads = {
            'dw1': dw1,
            'db1': db1,
            'dw2': dw2,
            'db2': db2
        }

    def update_gradients(self):
        self.params['w1'] = self.params['w1'] - self.grads['dw1'] * self.lr
        self.params['b1'] = self.params['b1'] - self.grads['db1'] * self.lr
        self.params['w2'] = self.params['w2'] - self.grads['dw2'] * self.lr
        self.params['b2'] = self.params['b2'] - self.grads['db2'] * self.lr

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.init_weights_and_bias()

        for i in range(self.epoch):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.update_gradients()
            self.loss.append(loss)

    def predict(self, x):
        z1 = x.dot(self.params['w1']) + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        pred = self.sigmoid(z2)

        return np.round(pred)

    def accuracy(self, y, yhat):
        acc = int(sum(y == yhat) / len(y) * 100)

        return acc
