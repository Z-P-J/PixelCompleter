import numpy as np


class OMP:
    """
    OMP算法封装成类
    """

    def __init__(self, dictionary, Y, k):
        """
        :param dictionary 字典
        :param Y 观测向量
        :param k 信号的稀疏度
        """
        self.dictionary = dictionary
        self.Y = Y
        self.T = k

    def start(self):
        """
        开始算法
        """
        if len(self.dictionary.shape) > 1:
            K = self.dictionary.shape[1]
        else:
            K = 1
            self.dictionary = self.dictionary.reshape((self.dictionary.shape[0], 1))
        if len(self.Y.shape) > 1:
            N = self.Y.shape[1]
        else:
            N = 1
            self.Y = self.Y.reshape((self.Y.shape[0], 1))
        X = np.zeros((K, N))
        for i in range(N):
            y = self.Y[:, i]
            # 初始化残差为y
            r = y
            index = []
            A = None
            x = None
            # 迭代
            for k in range(self.T):
                proj = np.fabs(np.dot(self.dictionary.T, r))
                # 最大投影系数对应的位置
                pos = np.argmax(proj)
                index.append(pos)
                # 更新索引集
                if k == 0:
                    A = self.dictionary[:, pos].reshape(self.Y.shape[0], 1)
                else:
                    A = np.concatenate((A, self.dictionary[:, pos].reshape(self.Y.shape[0], 1)), axis=1)
                # 最小二乘求得近似解
                x = np.dot(np.linalg.pinv(A), y)
                # 更新残差
                r = y - np.dot(A, x)

            tmp = np.zeros((K, 1))
            tmp[index] = x.reshape((self.T, 1))
            tmp = np.array(tmp).reshape(K)
            X[:, i] = tmp
        return X
