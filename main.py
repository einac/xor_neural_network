import numpy as np
from NeuralNetwork import *

if __name__ == '__main__':
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(layers=[2, 3, 1], lr=0.01, epoch=10000)
    nn.fit(inputs, outputs)

    train_preds = nn.predict(inputs)

    print("Predictions for train data: {}".format(train_preds))
    print("Train Accuracy: {} %".format(nn.accuracy(outputs, train_preds)))

    xtest = np.array([[0, 0]])
    ytest = np.array([[0]])

    test_preds = nn.predict(xtest)

    print()
    print("Predictions for test data: {}".format(test_preds))
    print("Test Accuracy: {} %".format(nn.accuracy(ytest, test_preds)))
