from model import SVMclassifier
# from SVM import SupportVectorMachine, SVM_one2all
from data_loader import MNIST, CIFAR10
import tensorboardX
import numpy as np
import time


def runMNIST():
    X_train, y_train = MNIST('./data/MNIST/', group='train')
    X_test, y_test = MNIST('./data/MNIST/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
    MNISTwriter = tensorboardX.SummaryWriter('./logs/mnist/kernel')



    # model = SVMclassifier(C=1.0)
    # start = time.time()
    # model.fit(X_train, y_train)
    # end = time.time()
    # print('train Time: {}'.format(end - start))
    # MNISTwriter.add_scalar(' MNIST train Time', end - start)

    # start  = time.time()
    # y_pred = model.predict(X_test)
    # end = time.time()
    # print('test Time: {}'.format(end - start))
    # MNISTwriter.add_scalar(' MNIST test Time', end - start)
    # accuracy = np.mean(y_pred == y_test)
    # print('Accuracy: {}'.format(accuracy))
    # MNISTwriter.add_scalar(' MNIST Accuracy', accuracy)


    kernel_list = ['sigmoid']
    for i, kernel in enumerate(kernel_list):
        model = SVMclassifier(kernel=kernel, gamma = 0.5 ** 10)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print('Time: {}'.format(end - start))
        MNISTwriter.add_scalar(' MNIST train Time with multi kernel', end - start, i)
        
        start  = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        print('Time: {}'.format(end - start))
        MNISTwriter.add_scalar(' MNIST test Time with multi kernel', end - start, i)
        accuracy = np.mean(y_pred == y_test)
        print('Accuracy: {}'.format(accuracy))
        MNISTwriter.add_scalar('MNIST Accuracy', accuracy, i)

def runCIFAR():
    X_train, y_train = CIFAR10('./data/cifar-10-batches-py/', group='train')
    X_test, y_test = CIFAR10('./data/cifar-10-batches-py/', group='test')
    X_train, X_test = X_train.reshape(-1, 3*32*32), X_test.reshape(-1, 3*32*32)
    CIFARwriter = tensorboardX.SummaryWriter('./logs/cifar/new')



    model = SVMclassifier(C=1.0)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('TRAIN Time: {}'.format(end - start))
    CIFARwriter.add_scalar(' CIFAR  train Time', end - start)

    start  = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    print('TEST Time: {}'.format(end - start))
    CIFARwriter.add_scalar(' CIFAR TEST Time', end - start)
    accuracy = np.mean(y_pred == y_test)
    print('Accuracy: {}'.format(accuracy))
    CIFARwriter.add_scalar(' MNIST Accuracy', accuracy)

    penalty_list = [0.5 ** i for i in range(0, 8)]
    for C in penalty_list:
        model = SVMclassifier(C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print('Accuracy: {}'.format(accuracy))
        CIFARwriter.add_scalar('CIFAR Accuracy with C', accuracy, C)

    gamma_list = [0.5 ** i for i in range(0, 4)]
    for gamma in gamma_list:
        model = SVMclassifier(gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print('Accuracy: {}'.format(accuracy))
        CIFARwriter.add_scalar('CIFAR Accuracy with gamma', accuracy, gamma)

    kernel_list = ['linear', 'polynomial', 'gaussian', 'sigmoid']
    for kernel in kernel_list:
        model = SVMclassifier(kernel=kernel)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print('train Time: {}'.format(end - start))
        CIFARwriter.add_scalar(' CIFAR TRAIN Time with multi kernel', end - start, kernel)
        
        start  = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        print('test Time: {}'.format(end - start))
        CIFARwriter.add_scalar(' CIFAR TEST Time with multi kernel', end - start, kernel)
        accuracy = np.mean(y_pred == y_test)
        print('Accuracy: {}'.format(accuracy))
        CIFARwriter.add_scalar('CIFAR Accuracy with multi kernel', accuracy, kernel)
runMNIST()
# runCIFAR()