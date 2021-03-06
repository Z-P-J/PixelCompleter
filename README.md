# PixelCompleter
 K-SVD算法和OMP算法实现人脸图像的像素缺失填补。主要参考文章[k-svd实现人脸缺失像素补全](https://blog.csdn.net/anonymity_/article/details/85196505)

# 实验说明
实验说明：
1.	使用K-SVD算法实现人脸图像的像素缺失填补实验，实验包括：

       1.1.	将数据集划分为训练集和测试集；
 
       1.2.	在训练集上使用K-SVD算法得到字典，字典大小自行设计或参考课程讲义；
 
       1.3.	对测试集中的人脸图像进行像素缺失处理；
 
       1.4.	在测试集上利用字典得到稀疏表达，然后使用字典重建人脸图像；
 
       1.5.	计算重建人脸图像与原始图像的平均误差。
       
2.	K-SVD算法主要包括：①初始化字典；②稀疏编码（OMP或BP算法）；③更新字典（通过SVD逐列更新）。
3.	需要参考表达学习课程讲义自己动手实现OMP算法和K-SVD算法流程，不能直接使用工具包。
4.	编程语言不限 

# 数据集
Yale B 数据集，38张192*168的人脸正面图像

# 如何使用
```python
from pixel_completer import PixelCompleter
PixelCompleter()\
        .set_data_set_dir('您的数据集目录')\
        .set_loss_img_dir('您的像素缺失图片保存目录')\
        .set_reconstruct_img_dir('您的像素缺失图片重建图片保存目录')\
        .set_loss_rate(50)\
        .set_iter_times(30)\
        .start()
```

# 结果
![像素缺失](https://raw.githubusercontent.com/Z-P-J/PixelCompleter/master/LossImg/loss0.jpg)
![重建](https://raw.githubusercontent.com/Z-P-J/PixelCompleter/master/RecImg/rec0.jpg)
![像素缺失](https://raw.githubusercontent.com/Z-P-J/PixelCompleter/master/LossImg/loss1.jpg)
![重建](https://raw.githubusercontent.com/Z-P-J/PixelCompleter/master/RecImg/rec1.jpg)

# 参考资料
[1]. k-svd实现人脸缺失像素补全https://blog.csdn.net/anonymity_/article/details/85196505

[2]. Robust Face Recognition via Sparse Representation https://ieeexplore.ieee.org/document/4483511

[3]. 基于压缩传感的匹配追踪重建算法研究 高睿

[4]. 字典学习（Dictionary Learning, KSVD）详解 https://www.cnblogs.com/endlesscoding/p/10090866.html

[5]. 稀疏表示（二）——KSVD算法详解（结合代码和算法思路）https://blog.csdn.net/tongdanping/article/details/79170375

[6]. 压缩感知重构算法之OMP算法python实现 https://blog.csdn.net/hjxzb/article/details/50923158

[7]. 稀疏表示去噪的理解 https://blog.csdn.net/tongdanping/article/details/79162547



