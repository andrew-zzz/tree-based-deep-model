from ximalaya_brain_jobs.train.vip.tdm.sample_init import data_process, tree_generate_samples, sample_merge_multiprocess, DataInput , map_generate
from ximalaya_brain_jobs.train.vip.tdm.construct_tree import TreeLearning
from ximalaya_brain_jobs.train.vip.tdm.din_model import Model
import tensorflow as tf
import pickle
import os
import sys
import time
import random
from ximalaya_brain_jobs.train.vip.tdm.dataset import DataGenerator
import pandas as pd
from ximalaya_brain_jobs.model.util import get_train_test_steps_dir


def get_data():
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/sample.pkl', 'rb') as f:
        data_train = pickle.load(f)
        data_validate = pickle.load(f)
        cache = pickle.load(f)
        return data_train, data_validate, cache

def run(model,train_set,test_set,model_save_path,train_step, validation_step):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      # print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
      lr = 0.1
      start_time = time.time()
      best_auc = 0.0
      for _ in range(1):
        loss_sum = 0.0
        for i in range(train_step):
          loss = model.train(sess, train_set, lr)
          loss_sum += loss
          if model.global_step.eval() % (train_step//4) == 0:
            auc = model._eval(sess, model,test_set,validation_step)
            print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
                  (model.global_epoch_step.eval(), model.global_step.eval(),
                   loss_sum / 1000, auc))
            if best_auc < auc:
                best_auc = auc
                model.save(sess, model_save_path)
            sys.stdout.flush()
            loss_sum = 0.0
          if model.global_step.eval() % 3000 == 0:
            lr = 0.01
        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time()-start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()

      print('best test_gauc:', best_auc)
      sys.stdout.flush()

def main():
    data_train, data_validate, cache = get_data()
    print('data_train len %d'% len(data_train))
    print('data_validate len %d' % len(data_validate))
    # uid,ts,item_list,behavior_list + mask
    _, _, tree = cache
    item_ids, item_size ,node_size = tree.items, len(tree.items),tree.node_size
    print('item_size %d' % item_size)
    print('node_size %d' % node_size)
    model = Model(item_size, node_size)
    num_epoch = 1
    model_save_path = '/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt'
    train_dir = '/home/dev/data/andrew.zhu/tdm/data_flow/train'
    test_dir = '/home/dev/data/andrew.zhu/tdm/data_flow/test'
    feature_dim = 4
    batch_size = 20000
    while num_epoch > 0:
        # ['item_ID', 'node', 'is_leaf', 'label']
        #生成树样本 把二叉树提出来用数组存储 提高访问效率
        node_list = tree._node_list(tree.root)
        print('node_list_len %d' % len(node_list))
        item_ids, item_size, node_size = tree.items, len(tree.items), tree.node_size
        start = time.clock()
        #根据树生成正负样本
        tree_samples = tree_generate_samples(item_ids, tree.leaf_dict, node_list)
        o = time.clock()
        print('finish tree_samples %d' % (o-start))
        print('tree_sample length %d' % len(tree_samples))
        #优化数据结构生成map
        s = time.clock()
        tree_map = map_generate(tree_samples)
        o = time.clock()
        print('finish build map %f' % (o-s))
        # 生成训练样本,文件写csv
        sample_merge_multiprocess(data_train , tree_map,'train',7,train_dir)
        sample_merge_multiprocess(data_validate,tree_map,'test',5,test_dir)
        # 获取训练数据,测试数据样本数
        train_step, validation_step = get_train_test_steps_dir(train_dir, test_dir, batch_size)
        print('train_step:%d' % train_step)
        print('test_step:%d' % validation_step)
        data_generator = DataGenerator(train_dir, test_dir, feature_dim, batch_size=batch_size)
        train_dataset,test_dataset = data_generator.datasetCreate()
        train = train_dataset.make_one_shot_iterator().get_next()
        test = test_dataset.make_one_shot_iterator().get_next()
        run(model,train,test,model_save_path,train_step,validation_step)
        num_epoch -= 1

        if num_epoch > 0:
            item_embeddings = model.get_embeddings(item_ids,model_save_path)
            tree = TreeLearning(item_embeddings, item_ids)
            _ = tree.clustering_binary_tree()
            tree._rebuild_item_list()
            with open('/home/dev/data/andrew.zhu/tdm/data_flow/final_tree.pkl', 'wb') as f:
                pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
    # dtest = Dataset(vtrain, 100)
    # metrics_count(dtest, tree.root, 150, model)
    print("========================================== end ==========================================")


