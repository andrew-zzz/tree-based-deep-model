from .din_model import Model
import pickle
from .tdm import get_data
import numpy as np
from ximalaya_brain_jobs.model.util import upload_result_to_hdfs


def load_data():
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/final_tree.pkl', 'wb') as f:
        tree = pickle.load(f)
        return tree

def embedding_index(embeddings, space='ip'):
    """
    通过 hnswlib 建立 item向量索引，从而快速进行最近邻查找
    :param sess:
    :param space:
    :return:
    """
    # embeddings = sess.run("vec/clip_V:0")
    print("embeddings type is %s" % type(embeddings))
    dim = embeddings.shape[1]
    print('embeddings shape is %s' % str(embeddings.shape))

    # # 建立索引
    import hnswlib
    nmsl_index = hnswlib.Index(space=space, dim=dim)
    nmsl_index.init_index(max_elements=100000, ef_construction=200)
    nmsl_index.set_ef(50)
    nmsl_index.add_items(embeddings)
    return nmsl_index

def get_embedding():
    data_train, data_validate, cache = get_data()
    print('data_train len %d'% len(data_train))
    print('data_validate len %d' % len(data_validate))
    # uid,ts,item_list,behavior_list + mask
    _, _, tree = cache
    item_ids, item_size ,node_size = tree.items, len(tree.items),tree.node_size
    print('item_size %d' % item_size)
    print('node_size %d' % node_size)
    model = Model(item_size, node_size)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt")
        item_embeddings = sess.run(model.item_emb_w)
        # print(item_embeddings.tolist())
        return np.array(item_embeddings)

        # print(item_embeddings.tolist())
        # return item_embeddings

def get_dict():
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/sample.pkl', 'rb') as f:
        data_train = pickle.load(f)
        data_validate = pickle.load(f)
        cache = pickle.load(f)
        return cache

def get_item_similar_item(index_item_dict, nmsl, embeddings, save_path, file_name, topK=30):
    """
    获取item相似item
    :param index_album_dict:
    :param nmsl:
    :param embeddings:
    :param topK:
    :return:
    """
    print("top k is %s" % topK)

    labels, distance = nmsl.knn_query(embeddings, k=topK)
    print(labels.shape)
    print(distance.shape)
    item_length = len(index_item_dict)
    print("sim album num is %d" % item_length)
    result = {}
    for i in range(item_length):
        # print(i)
        item_id = index_item_dict[i]
        label = labels[i]

        items = []
        for j in label.tolist():
            try:
                items.append(index_item_dict[int(j)])
            except:
                print('-- %s -- %s' % (j, type(j)))
        # albums = [index_album_dict[j] for j in label.tolist()]
        similar_item = []
        for k in range(topK):
            similar_item.append(str(items[k]))
        sim_item = '|'.join(similar_item)
        result[item_id] = sim_item
        # print("album_id is %s" % album_id)
        # print("sim album is %s" % sim_album)

    from pandas.core.frame import DataFrame
    re = DataFrame.from_dict(result, orient='index', columns=['re_items'])
    re = re.reset_index().rename(columns={'index': 'item_id'})
    # re.rename(columns={0: 'album_id', 1: 're_albums'}, inplace=True)
    print(re.head(5))
    re.to_csv(save_path + file_name, index=True)
    upload_result_to_hdfs("/user/dev/andrew.zhu/test",
                          save_path + file_name)
    return result

import tensorflow as tf

def main():
    # tree = load_data()
    # save_path = '/home/dev/data/andrew.zhu/tdm/model/tdm.ckpt'
    # item_ids, item_size, node_size = tree.items, len(tree.items), tree.node_size

    item_embedding = get_embedding()
    item_embedding_index = embedding_index(item_embedding)
    #
    (user_dict, item_dict, random_tree) = get_dict()
    item_sim_item_save_path='/home/dev/data/andrew.zhu/tdm/data_flow/'
    file_name='album_sim'
    item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    get_item_similar_item(item_dict, item_embedding_index, item_embedding, item_sim_item_save_path, file_name, 3)