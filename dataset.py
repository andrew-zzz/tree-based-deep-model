import tensorflow as tf
import os
import numpy as np

class DataGenerator(object):
    def __init__(self, train_dir, test_dir, feature_dim, batch_size=20000):
        """
        读取特定格式的csv文件，生成训练数据和测试数据
        :param train_dir: 训练数据路径
        :param test_dir: 测试数据路径
        :param feature_dim: 原始特征总数
        :param batch_size: batch大小
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.feature_dim = feature_dim
        self.batch_size = batch_size

    def parse_csv(self, value):
        """
        将csv行解析为tensor对象
        :param value:
        :return:
        """
        #uid 0
        #play_list 1-5
        #list_len  6
        #node_id   7
        #is_leaf   8
        #label     9
        columns = tf.decode_csv(value, record_defaults=[
            [0.0], [0.0], [0.0], [0.0], [0.0],
            [0.0], [0.0], [0.0], [0.0], [0.0]
        ])
        features = columns[0:9]
        label = columns[9]
        return features,label

    def datasetCreate(self):
        """
        创建train、test数据的dataset
        :return:
        """
        train_filenames = [self.train_dir +'/'+ filename for filename in os.listdir(self.train_dir)]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
        train_dataset = train_dataset.flat_map(
            lambda filename:
            tf.data.TextLineDataset(filename).map(self.parse_csv)
            # tf.data.TextLineDataset(filename).skip(1).apply(self.parse_csv)
        ).batch(self.batch_size).repeat().prefetch(100)
        #
        test_filenames = [self.train_dir +'/'+ filename for filename in os.listdir(self.test_dir)]
        test_dataset = tf.data.Dataset.from_tensor_slices(test_filenames)
        test_dataset = test_dataset.flat_map(
            lambda filename:
            tf.data.TextLineDataset(filename).map(self.parse_csv)
        ).batch(self.batch_size).repeat().prefetch(100)
        return train_dataset,test_dataset

