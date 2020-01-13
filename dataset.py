import tensorflow as tf
import os
import numpy as np
import linecache
import math

def get_train_test_steps_dir(deep_train_dirs, deep_test_dirs, batch_size):
    '''
    打印训练和测试数据的基本信息，获取训练和测试的步数
    :param deep_train_dirs:
    :param deep_test_dirs:
    :param batch_size
    :return:
    '''
    train_data_num = 0
    if os.path.exists(deep_train_dirs):
        for filename in os.listdir(deep_train_dirs):
            if filename=="_SUCCESS":
                continue
            full_filepath = os.path.join(deep_train_dirs, filename)
            file_lines = len(linecache.getlines(full_filepath)) - 1
            train_data_num = train_data_num + file_lines
    print("train_data_num")
    print(train_data_num)
    test_data_num = 0
    if os.path.exists(deep_test_dirs):
        for filename in os.listdir(deep_test_dirs):
            if filename=="_SUCCESS":
                continue
            full_filepath = os.path.join(deep_test_dirs, filename)
            file_lines = len(linecache.getlines(full_filepath)) - 1
            test_data_num = test_data_num + file_lines
    train_steps = math.ceil(train_data_num / batch_size)
    validation_steps = math.ceil(test_data_num / batch_size)
    linecache.clearcache()
    return train_steps, validation_steps

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

