from model import SVMclassifier
from data_loader import MNIST, CIFAR10
import tensorboardX
import numpy as np
import time

X_train, y_train = MNIST('./data/MNIST/', group='train')
X_test, y_test = MNIST('./data/MNIST/', group='test')
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)


# writer = tensorboardX.SummaryWriter('./logs/mnist/ablation/linear_kernel')
# penalty_list = [0.5 ** i for i in range(0, 8)]
# for i, C in enumerate(penalty_list):
#     model = SVMclassifier(C=C, kernel = 'linear')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     print('Accuracy: {}'.format(accuracy))
#     writer.add_scalar('MNIST Accuracy', accuracy, i)

# writerr = tensorboardX.SummaryWriter('./logs/mnist/ablation/rbf_kernel')
# gamma_list = [0.5 ** i for i in range(0, 8)]
# for index, gamma in enumerate(gamma_list):
#     model = SVMclassifier(gamma=gamma, kernel='rbf')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     print('Accuracy: {}'.format(accuracy))
#     writerr.add_scalar('MNIST Accuracy', accuracy, index)


# writerp = tensorboardX.SummaryWriter('./logs/mnist/ablation/poly_kernel')
# degree_list = [i for i in range(1, 8)]
# for index, degree in enumerate(degree_list):
#     model = SVMclassifier(d=degree, kernel='poly')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     print('Accuracy: {}'.format(accuracy))
#     writerp.add_scalar('MNIST Accuracy', accuracy, index)

writers = tensorboardX.SummaryWriter('./logs/mnist/ablation/rbf_kernel')
gamma_list = [0.5 ** i for i in range(0, 8)]
for index, gamma in enumerate(gamma_list):
    model = SVMclassifier(gamma = gamma, kernel='rbf')
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    test_time = end - start
    print('Train Time: {}'.format(train_time))
    print('Test Time: {}'.format(test_time))
    accuracy = np.mean(y_pred == y_test)
    print('Accuracy: {}'.format(accuracy))
    writers.add_scalar('MNIST Accuracy', accuracy, index)

