from data_set_manager import DataSetManager
from k_svd import KSVD
from omp import OMP
import numpy as np
import cv2
import  os


class PixelCompleter:
    """
    对像素补全功能封装成类
    """

    def __init__(self):
        # 数据集管理类
        self.data_set_manager = DataSetManager()
        # 缺失图片保存路径
        self.loss_img_dir = None
        # 重建图片保存路径
        self.reconstruct_img_dir = None
        # 像素缺失比例
        self.loss_rate = 50
        # 迭代次数
        self.iter_times = 30

    def set_data_set_dir(self, data_set_dir):
        """
        设置数据集路径
        :param data_set_dir 数据集路径
        """
        self.data_set_manager.set_data_set_dir(data_set_dir)
        if not os.path.exists(data_set_dir):
            os.makedirs(data_set_dir)
        return self

    def set_loss_img_dir(self, loss_img_dir):
        """
        设置像素缺失处理图片保存路径
        :param loss_img_dir 像素缺失处理图片保存路径
        """
        self.loss_img_dir = loss_img_dir
        if not os.path.exists(loss_img_dir):
            os.makedirs(loss_img_dir)
        return self

    def set_reconstruct_img_dir(self, reconstruct_img_dir):
        """
        设置重建人脸图像保存路径
        :param reconstruct_img_dir 重建的人脸图像保存路径
        """
        self.reconstruct_img_dir = reconstruct_img_dir
        if not os.path.exists(reconstruct_img_dir):
            os.makedirs(reconstruct_img_dir)
        return self

    def set_loss_rate(self, loss_rate):
        self.loss_rate = loss_rate
        return self

    def set_iter_times(self, iter_times):
        self.iter_times = iter_times
        return self


    def __merge_img(self, patchs, shape):
        """
        图像分为8*8的原子，纵向合并
        """
        img = np.zeros(shape)
        dim_r = img.shape[0] // 8
        dim_c = img.shape[1] // 8
        for i in range(dim_r):
            for j in range(dim_c):
                r = i * 8
                c = j * 8
                img[r:r + 8, c:c + 8] = patchs[:, i * dim_c + j].reshape(8, 8)
        return img

    def __loss(self, img):
        """
        像素缺失处理
        """
        missed_img = self.data_set_manager.miss_img_pixel(img, self.loss_rate)
        return self.__merge_img(missed_img, img.shape)

    def __reconstruct(self, img, dictionary, K):
        """
        使用字典重建人脸图像
        """
        patchs = self.data_set_manager.sep_img(img)
        for i in range(patchs.shape[1]):
            patch = patchs[:, i]
            index = np.nonzero(patch)[0]
            if index.shape[0] == 0:
                continue
            # l2归一化
            l2norm = np.linalg.norm(patch[index])
            mean = np.sum(patch) / index.shape[0]
            patch_norm = (patch - mean) / l2norm
            x = OMP(dictionary[index, :], patch_norm[index].T, K).start()
            patchs[:, i] = np.fabs(((dictionary.dot(x) * l2norm) + mean).reshape(patchs.shape[0]))
        return self.__merge_img(patchs, img.shape)

    def __psnr(self, a, b):
        """
        计算PSNR值
        """
        # if a.any() == b.any():
        #     return 0
        # else:
        return 10 * np.log10(a.shape[0] * a.shape[1] / (((a.astype(np.float) - b) ** 2).mean()))

    def start(self):
        """
        启动像素补全程序
        """
        print("Prepare Date set!")
        self.data_set_manager.prepare_set()
        print("Prepare Data set finished!")

        # 测试集
        test_set = self.data_set_manager.get_test_set()
        # 训练集
        train_img = self.data_set_manager.get_train_img()

        # 字典训练
        print("K-SVD fit start")
        dictionary = KSVD(self.iter_times, 256, 50).fit(train_img)
        print("K-SVD fit finished")

        for i in range(len(test_set)):
            # 对测试集中的人脸图像进行像素缺失处理
            print("图片" + str(i) + "像素缺失处理开始。")
            loss = self.__loss(test_set[i])
            # loss = self.__merge_img(self.data_set_manager.miss_img_pixel(test_set[i], 50), test_set[i].shape)
            loss_img_name = "loss" + str(i) + ".jpg"
            cv2.imwrite(self.loss_img_dir + loss_img_name, loss.astype(np.uint8))
            print("图片" + str(i) + "像素缺失处理完成。")
            # 在测试集上利用字典得到稀疏表达，然后使用字典重建人脸图像
            print("图片" + str(i) + "重建开始。")
            rec_img = self.__reconstruct(loss, dictionary, 256)
            rec_img_name = "rec" + str(i) + ".jpg"
            cv2.imwrite(self.reconstruct_img_dir + rec_img_name, rec_img.astype(np.uint8))
            print("图片" + str(i) + "重建完成。psnr=" + str(self.__psnr(test_set[i], rec_img)))
            # print("loss img " + str(i) + " is reconstructed!  psnr=" + str(self.__psnr(test_set[i], rec_img)))


if __name__ == "__main__":
    # 运行PixelCompleter
    PixelCompleter()\
        .set_data_set_dir('D:\\Programming\\YaleB\\DataSet\\')\
        .set_loss_img_dir('D:\\Programming\\YaleB\\LossImg\\')\
        .set_reconstruct_img_dir('D:\\Programming\\YaleB\\RecImg\\')\
        .set_loss_rate(50)\
        .set_iter_times(20)\
        .start()