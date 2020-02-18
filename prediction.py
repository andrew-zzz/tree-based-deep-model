import numpy as np
from ximalaya_brain_jobs.train.vip.tdm.tdm import get_data
from ximalaya_brain_jobs.model.util import upload_result_to_hdfs
from ximalaya_brain_jobs.train.vip.tdm.din_model import Model
import tensorflow as tf
import pickle
import sklearn.metrics as metrics

def make_data(state,node):
    length = len(state)
    r_val = [0]
    r_val.extend(state)
    r_val.append(length)
    if node.item_id is not None:
        r_val.append(node.item_id)
        r_val.append(1)
    else:
        r_val.append(node.val)
        r_val.append(0)
    return np.array([r_val])


def candidates_generator(state, root, k, model,sess):
    """layer-wise retrieval algorithm in prediction."""
    Q, A = [root], []
    time = 0
    while Q:
        Q_tmp = []
        for node in Q:
            if node.item_id is not None:
                A.append(node)
            else:
                Q_tmp.append(node)
        Q = Q_tmp
        probs = []
        for node in Q:
            data = make_data(state,node)
            prob = model.predict(data,sess)
            probs.append(prob[0][0])
        prob_list = list(zip(Q, probs))
        prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
        print('prob_list %d' % time)
        for i in prob_list:
            print('time:%d,id %d:score %f' %(time,i[0].val,i[1]))
        time = time + 1
        I = []
        for i in prob_list[0:k]:
            I.append(i[0])
        # print(I)
        Q = []
        for j in I:
            if node.left:
                Q.append(j.left)
            if node.right:
                Q.append(j.right)
        t = []
        for i in range(len(Q)):
            if Q[i].item_id == None:
                t.append(Q[i].val)
            else:
                t.append(Q[i].item_id)
    probs = []
    for leaf in A:
        data = make_data(state,leaf)
        prob = model.predict(data,sess)
        probs.append(prob[0])
    prob_list = list(zip(A, probs))
    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    A = []
    for i in range(k):
        A.append(prob_list[i][0].item_id)
    return A

def metrics_count(data, root, k, model):
    """Recall/Precision/F-measure statistic."""
    precision_rate, recall_rate, fm_rate, novelty_rate, num = 0, 0, 0, 0, 0
    for items in data:
        size = items.shape[0]
        for i in range(size):
            cands = candidates_generator((items[i][None, :],), root, k, model)
            item_clip = list(set(items[i][items[i] != -2].tolist()))
            m, g = len(cands), len(item_clip)
            for item in item_clip:
                if item in cands:
                    cands.remove(item)
            n = len(cands)
            p_rate, r_rate, n_rate = float(m - n) / m, float(m - n) / g, float(n) / k
            f_rate = (2 * p_rate * r_rate) / (p_rate + r_rate)
            precision_rate += p_rate
            recall_rate += r_rate
            fm_rate += f_rate
            novelty_rate += n_rate
            num += 1
    precision_rate = float(precision_rate * 100) / num
    recall_rate = float(recall_rate * 100) / num
    fm_rate = float(fm_rate * 100) / num
    novelty_rate = float(novelty_rate * 100) / num
    print("================================= Performance Statistic =================================")
    print("Precision rate: {:.2f}% | Recall rate: {:.2f}% | "
          "F-Measure rate: {:.2f}% | Novelty rate: {:.2f}%".format(precision_rate, recall_rate, fm_rate, novelty_rate))


def main():
    data_train, data_validate, cache = get_data()
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/final_tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    item_ids, item_size, node_size = tree.items, len(tree.items), tree.node_size
    print(item_size)
    print(node_size)
    model = Model(item_size, node_size,9)
    # play_hist = [29309729,23571206,28580191,321787,26565248]
    # play_hist_index = []
    # for i in play_hist:
    #     play_hist_index.append(item_index[i])
    play_hist_index = [100.0, 77.0, 800.0, 999.0,1200.0]

    # print(play_hist_index)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用 GPU
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt")
        # data = np.array([[12.0, 552.0, 853.0, 12.0, 283.0, 210.0, 5.0, 0.0, 0.0]])
        # import time
        # t1 = time.clock()
        # rval = model.predict(data, sess)
        # t2 = time.clock()
        # print(t2 - t1)
        # print('rval')
        # print(rval)
        import time
        ts = time.clock()
        result = candidates_generator(play_hist_index, tree.root, 50, model,sess)
        ts1 = time.clock()
        print(ts1 - ts)
        print(result)
        # result_albumId = []
        # for i in result:
        #     result_albumId.append(index_item[i])
        # print(result_albumId)
