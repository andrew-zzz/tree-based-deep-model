# 深度树模型实验
该项目优化和完善了git另外一个哥们写的实验室类型的的项目,优化了大数据量情况下在生产环境成功运行
---

### 代码结构
文件说明  
tdm.py: 代码入口，负责完整深度树模型的训练和测试  
sample_init.py: 数据处理及生成程序，负责数据预处理及树样本的生成  
construct_tree.py: 样本二叉树生成程序，负责树模型的生成  
din_model.py: DIN网络搭建
prediction.py: 遍历树预测部分
dataset.py:数据生成迭代器 
relative_album_caculate:专辑的相关专辑计算


### 算法模型
深度树算法流程(文献[1]):  
1. 构造随机二叉树  
2. 基于树模型生成样本  
3. 训练DNN模型直到收敛  
4. 基于DNN模型得到样本的Embedding，重新构造聚类二叉树  
5. 循环上述2～4过程

### 进度
100w用户,每个用户5个播放历史跑通

### 参考文献
    [1] Learning Tree-based Deep Model for Recommender Systems, Han Zhu, Xiang Li, Pengye Zhang, etc.
    [2] Deep Interest Network for Click-Through Rate Prediction, Guorui Zhou, Chengru Song, Xiaoqiang Zhu, etc.
    [3] Empirical Evaluation of Rectified Activations in Convolution Network, Bing Xu, Naiyan Wang, Tianqi Chen, etc.
    [4] Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, etc.
    [5] Distributed Representations of Words and Phrases and their Compositionality, Tomas Mikolov, Ilya Sutskever, Kai Chen, etc.
