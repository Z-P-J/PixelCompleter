import os
import cv2
import numpy as np


class DataSetManager:
    """
    数据集管理类
    """

    __data_set_dir = None
    __ratio = 0.8
    __data = []
    __data_num = 0
    __train_set = []
    __test_set = []
    __n = 504
    __train_img = None

    def set_data_set_dir(self, data_set_dir):
        """
        设置数据集路径
        :param data_set_dir 数据集路径
        """
        self.__data_set_dir = data_set_dir

    def prepare_set(self):
        # 读取数据集
        for root, dir, files in os.walk(self.__data_set_dir):
            for file in files:
                self.__data.append(cv2.imread(self.__data_set_dir + str(file), -1))
                self.__data_num += 1

        # 准备训练集和测试集
        random_num = np.random.randint(0, high = self.__data_num,
                                       size = int(self.__ratio * self.__data_num))
        for i in range(self.__data_num):
            if i not in random_num:
                self.__test_set.append(self.__data[i])
            else:
                self.__train_set.append(self.__data[i])

    def miss_img_pixel(self, img, k=50):
        """
        像素缺失处理
        :param img 测试样本
        :param k 像素缺失比例
        """
        patchs = self.sep_img(img)
        k = int(k * 0.01 * patchs.shape[0] * patchs.shape[1])
        loss_r = np.random.randint(0, high=patchs.shape[0] - 1, size=k)
        loss_c = np.random.randint(0, high=patchs.shape[1] - 1, size=k)
        for i in range(k):
            patchs[loss_r[i], loss_c[i]] = 0
        return patchs

    def sep_img(self, img):
        """
        图像分为8*8的原子，纵向合并
        """
        dim_r = img.shape[0] // 8
        dim_c = img.shape[1] // 8
        dim = dim_r * dim_c
        patchs = np.zeros((64, dim))
        for i in range(dim_r):
            for j in range(dim_c):
                r = i * 8
                c = j * 8
                patchs[:, i * dim_c + j] = img[r:r + 8, c:c + 8].reshape(64)
        return patchs

    def get_data_set(self):
        return self.__data

    def get_test_set(self):
        return self.__test_set

    def get_train_set(self):
        return self.__train_set

    def get_train_img(self):
        """
        获取训练图片
        """
        atoms = np.array(self.sep_img(self.__train_set[0]))
        for i in range(1, len(self.__train_set)):
            patchs = self.sep_img(self.__train_set[i])
            # 矩阵拼接
            atoms = np.concatenate((atoms, patchs), axis=1)
        # return atoms[:, np.random.randint(0, high=atoms.shape[1] - 1, size=self.__n)]
        return self.sep_img(atoms[:, np.random.randint(0, high=atoms.shape[1] - 1, size=self.__n)])
