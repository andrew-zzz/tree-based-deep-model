from .sample_init import data_process, tree_generate_samples, merge_samples, DataInput , map_generate
from .prediction import metrics_count
from .construct_tree import TreeLearning
from .din_model import Model
import tensorflow as tf
import pickle
import os
import sys
import time
import random
import pandas as pd

def get_data():
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/sample.pkl', 'rb') as f:
        data_train = pickle.load(f)
        data_validate = pickle.load(f)
        cache = pickle.load(f)
        return data_train, data_validate, cache

def run(model,train_set,test_set,train_batch_size,test_batch_size,model_save_path):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      # print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
      sys.stdout.flush()
      lr = 0.1
      start_time = time.time()
      best_auc = 0.0
      for _ in range(20):
        random.shuffle(train_set)
        # epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
          loss = model.train(sess, uij, lr)
          loss_sum += loss
          if model.global_step.eval() % 2000 == 0:
            auc = model._eval(sess, model,test_set,test_batch_size)
            print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
                  (model.global_epoch_step.eval(), model.global_step.eval(),
                   loss_sum / 1000, auc))
            if best_auc < auc:
                best_auc = auc
                model.save(sess, model_save_path)
            sys.stdout.flush()
            loss_sum = 0.0
          if model.global_step.eval() % 10000 == 0:
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
    print(data_train)
    print('data_validate len %d' % len(data_validate))
    # uid,ts,item_list,behavior_list + mask
    _, _, tree = cache
    item_ids, item_size ,node_size = tree.items, len(tree.items),tree.node_size
    print('item_size %d' % item_size)
    print('node_size %d' % node_size)
    model = Model(item_size, node_size)
    num_epoch = 3
    train_batch_size = 20000
    test_batch_size = 20000
    model_save_path = '/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt'
    while num_epoch > 0:
        # ['item_ID', 'node', 'is_leaf', 'label']
        #生成树样本
        node_list = tree._node_list(tree.root)
        print('node_list_len %d' % len(node_list))
        item_ids, item_size, node_size = tree.items, len(tree.items), tree.node_size
        start = time.clock()
        tree_samples = tree_generate_samples(item_ids, tree.leaf_dict, node_list)
        o = time.clock()
        print('finish tree_samples %d' % (o-start))
        print(len(tree_samples))
        #优化数据结构生成map
        s = time.clock()
        tree_map = map_generate(tree_samples)
        o = time.clock()
        print('finish build map %f' % (o-s))
        #生成训练样本
        # user_ID  item_ID timestamp behaviors  node  is_leaf  label
        # explode join待优化
        s = time.clock()
        merge_samples(data_train , tree_map ,"train"), merge_samples(data_validate ,tree_map,"test")

        tdata = pd.read_csv("/home/dev/data/andrew.zhu/tdm/data_flow/train.csv").values
        vdata = pd.read_csv("/home/dev/data/andrew.zhu/tdm/data_flow/test.csv").values
        o = time.clock()
        print('tdata %d %f' % (len(tdata),(o-s)))
        print('vdata %d' % len(vdata))
        tdata_val = tdata.values
        vdata_val = vdata.values
        run(model,tdata_val,vdata_val,train_batch_size,test_batch_size,model_save_path)
        num_epoch -= 1
        if num_epoch > 0:
            item_embeddings = model.get_embeddings(item_ids,model_save_path)
            tree = TreeLearning(item_embeddings, item_ids)
            _ = tree.clustering_binary_tree()
            with open('/home/dev/data/andrew.zhu/tdm/data_flow/final_tree.pkl', 'wb') as f:
                pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)  # uid, iid

    # dtest = Dataset(vtrain, 100)
    # metrics_count(dtest, tree.root, 150, model)
    print("========================================== end ==========================================")


