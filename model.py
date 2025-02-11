import numpy as np
import random
import time
import numpy as np
from tqdm import tqdm
class SimpleSVM:
    def __init__(self, C=1.0, lr=0.01, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.w)
                else:
                    self.w -= self.lr * (2 * self.w - np.dot(x_i, y_[idx]) * self.C)
                    self.b -= self.lr * y_[idx] * self.C

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class BinarySVM:
    def __init__(self, C: float = 1.0, kernel = None,
                 kernel_weight: np.ndarray = [0, 1, 0, 0], epsilon: float = 1e-6,maxIter: int = 40,
                 gamma: float = 1.0, c: float=0.0, 
                 d: float = 2) -> None:
        self.C = C
        self.gamma = gamma
        self.c = c
        self.d = d
        self.epsilon = epsilon
        self.maxIter = maxIter  
        self.kernel_weight = kernel_weight
        if kernel is None or kernel == 'linear':
            self.kernel = self.linear_kernel
        elif kernel == 'rbf':
            self.kernel = self.kernel_rbf
        elif kernel == 'poly':  
            self.kernel = self.kernel_poly
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid_kernel
        elif kernel == 'multi':
            self.kernel = self.multi_kerneal

        # print(C, kernel_weight, gamma, c, d)
    def __compute_w(self):
        return (self.a * self.y) @ self.X

    def __compute_e(self, i):
        return (self.a * self.y) @ self.K[:, i] + self.b - self.y[i]
    
    def __select_j(self, i):
        j = np.random.randint(1, self.m)
        return j if j > i else j - 1
    
    def __step_forward(self, i):
        e_i = self.__compute_e(i)
        if ((self.a[i] > 0) and (e_i * self.y[i] > self.epsilon)) or ((self.a[i] < self.C) and (e_i * self.y[i] < -self.epsilon)):
            j = self.__select_j(i)
            e_j = self.__compute_e(j)
            a_i, a_j = np.copy(self.a[i]), np.copy(self.a[j])
            if self.y[i] == self.y[j]:
                L = max(0, a_i + a_j - self.C)
                H = min(self.C, a_i + a_j)
            else:
                L = max(0, a_j - a_i)
                H = min(self.C, self.C + a_j - a_i)
            if L == H:
                return False
            d = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if d >= 0:
                return False
            self.a[j] = np.clip(a_j - self.y[j] * (e_i - e_j) / d, L, H)
            if np.abs(self.a[j] - a_j) < self.epsilon:
                return False
            self.a[i] = a_i + self.y[i] * self.y[j] * (a_j - self.a[j])
            b_i = self.b - e_i - self.y[i] * self.K[i, i] * (self.a[i] - a_i) - self.y[j] * self.K[j, i] * (self.a[j] - a_j)
            b_j = self.b - e_j - self.y[i] * self.K[i, j] * (self.a[i] - a_i) - self.y[j] * self.K[j, j] * (self.a[j] - a_j)
            if 0 < self.a[i] < self.C:
                self.b = b_i
            elif 0 < self.a[j] < self.C:
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2
            return True
        return False
    def SMO(self, x: np.ndarray, y: np.ndarray, tol: int = 1e-3, maxIter: int = 40):
        # m, n = x.shape
        # b = 0.0
        # alpha = np.zeros(m, dtype=np.float32)
        flag = True
        for i in range(self.maxIter):
            count = 0
            if flag:
                for i in range(self.m):
                    count += self.__step_forward(i)
            else:
                index = np.nonzero((0 < self.a) * (self.a < self.C))[0]
                for i in index:
                    count += self.__step_forward(i)
            if flag:
                flag = False
            elif count == 0:
                flag = True

    
    def linear_kernel(self, X: np.ndarray, y: np.ndarray):
        return X @ y.T
    
    def kernel_rbf(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(- self.gamma * ((np.linalg.norm((x1 - x2.T), axis=1)) ** 2))
    
    def kernel_poly(self, x1: np.ndarray, x2: np.ndarray):
        return (x1 @ x2.T + self.c) ** self.d
    
    def sigmoid_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.tanh(self.gamma * (x1 @ x2.T) + self.c)
    
    def multi_kerneal(self, x1: np.ndarray, x2: np.ndarray):
        m = x1.shape[-1]
        x1 = np.reshape(x1, (-1, m))
        return self.kernel_weight @ np.array([self.kernel_Linear(x1, x2), 
                                              self.kernel_rbf(x1, x2),
                                              self.kernel_poly(x1, x2), 
                                              self.sigmoid_kernel(x1, x2)])
    

    def fit(self, data: np.ndarray, label: np.ndarray):
        ti = time.time()
        self.m, self.n = data.shape
        self.X = data
        self.y = label
        # self.alpha = np.zeros(self.m, dtype=np.float32)
        self.b = 0.0
        self.a = np.zeros(self.m)

        self.K = np.zeros((self.m, self.m), dtype=np.float32)
        for i in range(self.m):
            self.K[i, :] = self.kernel(self.X, self.X[i])

        self.SMO(self.X, self.y)


    def predict(self, X: np.ndarray):
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred[i] = np.sum(self.a * self.y * self.kernel(self.X, X[i])) + self.b
        return np.sign(pred)

    def select(self, i: int, m: int):
        j = random.choice(range(m))
        while (j == i):
            j = random.choice(range(m))
        return j
    
class SVMclassifier(BinarySVM):
    def __init__(self, C: float = 1.0, kernel = None,
                 kernel_weight: np.ndarray = [0, 1, 0], epsilon: float = 1e-6,maxIter: int = 40,
                 gamma: float = 1.0, c: float=1.0, 
                 d: float = 2) -> None:
        self.C = C
        self.gamma = gamma
        self.maxIter = maxIter
        self.c = c
        self.d = d
        self.epsilon = epsilon
        self.kernel_weight = kernel_weight
        self.kernel = kernel
        self.classifier = []

    def fit(self, data: np.ndarray, label: np.ndarray):
        self.label = np.unique(label)
        for i in range(len(self.label)):
            for j in range(i+1, len(self.label)):
                model = BinarySVM(self.C, self.kernel, self.kernel_weight, self.epsilon, self.maxIter, self.gamma, self.c, self.d)
                # model = SupportVectorMachine(100, self.C, self.epsilon)
                # model.fit(data, np.where(label == self.label[i], 1, -1))
                self.classifier.append((i, j, model))
        for i, j, model, in tqdm(self.classifier):
            mask = np.where((label == self.label[i]) | (label == self.label[j]))[0]
            X = data[mask]
            y = np.where(label[mask] == self.label[i], -1, 1)
            model.fit(X, y)

    def predict(self, X: np.ndarray):
        pred = np.zeros((X.shape[0], len(self.label)))
        for i, j, model in tqdm(self.classifier):
            y = model.predict(X)
            pred[np.where(y == -1)[0], i] += 1
            pred[np.where(y == 1)[0], j] += 1
        return self.label[np.argmax(pred, axis=1)]
    

class MultikernelSVM(BinarySVM):
    def init(self, C: float = 1.0, kernel = None,
                 kernel_weight: np.ndarray = [0, 1, 0, 0], epsilon: float = 1e-6,maxIter: int = 40,
                 gamma: float = 1.0, c: float=1.0, 
                 d: float = 2) -> None:
        self.C = C
        self.gamma = gamma
        self.maxIter = maxIter
        self.c = c
        self.d = d
        self.epsilon = epsilon
        self.kernel_weight = kernel_weight
        self.component  = []
        for entry in kernel:
            if entry == 'linear':
                self.component.append(self.linear_kernel)
            elif entry == 'rbf':
                self.component.append(self.kernel_rbf)
            elif entry == 'poly':
                self.component.append(self.kernel_poly)
            elif entry == 'sigmoid':
                self.component.append(self.sigmoid_kernel)
        

    def kernel(self, X: np.ndarray, y: np.ndarray):
        return np.sum([self.w[i] * self.component[i](X, y) for i in range(self.w)], axis=0)
    
    def SimilarityMetrix(self, X: np.ndarray, Y: np.ndarray):
        return np.sum(X * Y)/ np.sqrt(np.sum(X* X) * np.sum(Y* Y))
    def fit(self, X, y):
        self.X, self.y = X, y
        self.m, self.n = X.shape
        self.b = 0.0
        self.a = np.zeros(self.m)
        num_kernel = len(self.component)
        self.w = np.zeros(num_kernel)
        self.K = np.zeros((num_kernel, self.m, self.m))
        for i in range(num_kernel):
            for j in range(self.m):
                self.K[i, :, j] = self.component[i](X, X[j, :])
            self.w[i] = self.SimilarityMatrix(self.K[i], np.outer(y, y))

        self.w = self.w / np.sum(self.w)
        self.K = np.sum([self.w[i] * self.K[i] for i in range(num_kernel)], axis=0)

        self.SMO(self.X, self.y)