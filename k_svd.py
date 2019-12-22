import numpy as np
from omp import OMP


class KSVD:

    """
    K-SVD算法
    """

    def __init__(self, max_iter_times, n_components, T, n = 64, sigma = 1e-6):

        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter_times: 最大迭代次数
        :param sigma: 稀疏表示结果的容差
        :param T: 稀疏度
        :param n: 原子维度
         """
        self.max_iter_times = max_iter_times
        self.n_components = n_components
        self.T = T
        self.sigma = sigma
        self.n = n
        # 生成初始字典
        self.dictionary = np.random.random((self.n, self.n_components))

    def __initialize(self, Y):

        """
        随机选取样本集Y中n_components个样本,并做L2归一化
        """

        for i in range(self.n_components):
            norm = np.linalg.norm(self.dictionary[:, i])
            mean = np.sum(self.dictionary[:, i]) / self.dictionary.shape[0]
            self.dictionary[:, i] = (self.dictionary[:, i] - mean) / norm

        for i in range(Y.shape[1]):
            norm = np.linalg.norm(Y[:, i])
            mean = np.sum(Y[:, i]) / Y.shape[0]
            Y[:, i] = (Y[:, i] - mean) / norm

    def __update_dic(self, X, Y, d):
        """
        使用K-SVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(X[i, :])[0]
            if len(index) == 0:
                continue
            # 更新第i列
            d[:, i] = 0
            # 计算误差矩阵
            R = (Y - np.dot(d, X))[:, index]
            # 利用svd的方法，来求解更新字典和稀疏系数矩阵
            u, s, v = np.linalg.svd(R, full_matrices=False)
            # 使用左奇异矩阵的第0列更新字典
            d[:, i] = u[:, 0].T
            # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
            X[i, index] = s[0] * v[0, :]

    def fit(self, Y):
        """
        K-SVD迭代过程
        """
        self.__initialize(Y)
        total = 0
        count = 0
        for j in range(self.max_iter_times):
            count = j + 1
            # OMP算法
            X = OMP(self.dictionary, Y, self.T).start()
            # 计算误差
            e = np.linalg.norm(Y - np.dot(self.dictionary, X))
            total = total + e
            print(str('第%s次迭代 误差=%s' % (j, e)) + '\n')
            # 如果误差小于稀疏表示结果的容差，则跳出迭代
            if e < self.sigma:
                break
            # 更新字典
            self.__update_dic(X, Y, self.dictionary)
        print("平均误差=" + str(total / count))
        return self.dictionary