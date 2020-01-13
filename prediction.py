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
    while Q:
        for node in Q:
            if node.item_id is not None:
                A.append(node)
                Q.remove(node)
        probs = []
        for node in Q:
            data = make_data(state,node)
            prob = model.predict(data,sess)
            print(prob)
            probs.append(prob[0])
        prob_list = list(zip(Q, probs))
        prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
        I = []
        if len(prob_list) > k:
            for i in range(k):
                I.append(prob_list[i][0])
        else:
            for p in prob_list:
                I.append(p[0])
        Q = []
        while I:
            node = I.pop()
            if node.left:
                Q.append(node.left)
            if node.right:
                Q.append(node.right)
    probs = []

    for leaf in A:
        data = make_data(state,leaf)
        prob = model.predict(data,sess)
        print(prob)
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
    user_dict, item_index, tree = cache
    item_ids, item_size, node_size = tree.items, len(tree.items), tree.node_size
    index_item = dict(zip(item_index.values(), item_index.keys()))

    model = Model(item_size, node_size)
    play_hist = [29309729,23571206,28580191,321787,26565248]
    play_hist_index = []
    for i in play_hist:
        play_hist_index.append(item_index[i])

    print(play_hist_index)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt")
        result = candidates_generator(play_hist_index, tree.root, 20, model,sess)
        result_albumId = []
        for i in result:
            result_albumId.append(index_item[i])
        print(result_albumId)
        # print(item_embeddings.tolist())
        # return np.array(item_embeddings)
