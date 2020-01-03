import os
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
from construct_tree import TreeInitialize
import copy

LOAD_DIR = os.path.dirname(os.path.abspath(__file__)) + '/datasets/UserBehavior_sp.csv'


def _time_window_stamp():
    boundaries = ['2017-11-26 00:00:00', '2017-11-27 00:00:00', '2017-11-28 00:00:00',
                  '2017-11-29 00:00:00', '2017-11-30 00:00:00', '2017-12-01 00:00:00',
                  '2017-12-02 00:00:00', '2017-12-03 00:00:00', '2017-12-04 00:00:00']
    for i in range(len(boundaries)):
        time_array = time.strptime(boundaries[i], "%Y-%m-%d %H:%M:%S")
        time_stamp = int(time.mktime(time_array))
        boundaries[i] = time_stamp
    return boundaries


def _time_converter(x, boundaries):
    tag = -1
    if x > boundaries[-1]:
        tag = 9
    else:
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
                tag = i
                break
    return tag


def _mask_padding(data, max_len):
    size = data.shape[0]
    raw = data.values
    mask = np.array([[-2] * max_len for _ in range(size)])
    for i in range(size):
        mask[i, :len(raw[i])] = raw[i]
    return mask.tolist()


def data_process():
    """convert and split the raw data."""
    #user_id,item_id,category_id,behavior_type index化
    data_raw = pd.read_csv(LOAD_DIR, header=None,
                           names=['user_ID', 'item_ID', 'category_ID', 'behavior_type', 'timestamp'])
    data_raw = data_raw[:10000]
    user_list = data_raw.user_ID.drop_duplicates().to_list()
    user_dict = dict(zip(user_list, range(len(user_list))))
    data_raw['user_ID'] = data_raw.user_ID.apply(lambda x: user_dict[x])
    item_list = data_raw.item_ID.drop_duplicates().to_list()
    item_dict = dict(zip(item_list, range(len(item_list))))
    data_raw['item_ID'] = data_raw.item_ID.apply(lambda x: item_dict[x])
    category_list = data_raw.category_ID.drop_duplicates().to_list()
    category_dict = dict(zip(category_list, range(len(category_list))))
    data_raw['category_ID'] = data_raw.category_ID.apply(lambda x: category_dict[x])
    behavior_dict = dict(zip(['pv', 'buy', 'cart', 'fav'], range(4)))
    data_raw['behavior_type'] = data_raw.behavior_type.apply(lambda x: behavior_dict[x])

    time_window = _time_window_stamp()
    #日期按照时间日 分成9天 0-9
    data_raw['timestamp'] = data_raw.timestamp.apply(_time_converter, args=(time_window,))

    #建立二叉树
    random_tree = TreeInitialize(data_raw)
    _ = random_tree.random_binary_tree()

    #行为数据按user_id,timestamp聚合
    data = data_raw.groupby(['user_ID', 'timestamp'])['item_ID'].apply(list).reset_index()
    data['behaviors'] = data_raw.groupby(['user_ID',
                                         'timestamp'])['behavior_type'].apply(list).reset_index()['behavior_type']
    data['behavior_num'] = data.behaviors.apply(lambda x: len(x))

    #过滤行为数据小于10次的user
    mask_length = data.behavior_num.max()
    data = data[data.behavior_num >= 10]
    data = data.drop(columns=['behavior_num'])

    #加mask
    data['item_ID'] = _mask_padding(data['item_ID'], mask_length)
    data['behaviors'] = _mask_padding(data['behaviors'], mask_length)
    #data 'user_ID',timestamp 'item_list', 'behaviors_list'
    data_train, data_validate, data_test = data[:-200], data[-200:-100], data[-100:]
    cache = (user_dict, item_dict, behavior_dict, random_tree)
    return data_train, data_validate.reset_index(drop=True), data_test.reset_index(drop=True), cache


def _single_node_sample_1(item_id, node, node_list):
    samples = []
    positive_info = []
    i = 0
    s = time.clock()
    while node:
        if node.item_id is None:
            single_sample = [item_id, node.val, 0, 1]
            id = node.val
        else:
            single_sample = [item_id, node.item_id, 1, 1]
            id = node.item_id
        samples.append(single_sample)
        positive_info.append(id)
        node = node.parent
        i += 1
    #j从root节点到叶子节点的index k代表当前level
    j, k = i-2, 1
    #当前tree_list_map数据结构为[[(id,is_leaf)],[]]
    tree_depth = len(node_list)
    for i in range(1,tree_depth):
        tmp_map = node_list[i]
        if len(tmp_map) <= 2*k:
            index_list = range(len(tmp_map))
        else:
            index_list = random.sample(range(len(tmp_map)), 2 * k)
        if j == 0:
            index_list = random.sample(range(len(tmp_map)), 3)
            remove_item = (positive_info[j], 1)
        else:
            remove_item = (positive_info[j], 0)
        sample_iter = []
        for level_index in index_list:
            single_sample = [item_id, tmp_map[level_index][0], tmp_map[level_index][1], 0]
            sample_iter.append(single_sample)

        if [item_id, remove_item[0], remove_item[1], 0] in sample_iter:
            sample_iter.remove([item_id, remove_item[0], remove_item[1], 0])
        samples.extend(sample_iter)
        k += 1
        j -= 1
        if(j < 0):break
    e = time.clock()
    print('finish %f' % (e-s))
    samples = pd.DataFrame(samples, columns=['item_ID', 'node', 'is_leaf', 'label'])
    return samples

def _single_node_sample(item_id, node, root):
    s = time.clock()
    sample_num = 300
    samples = []
    positive_info = {}
    i = 0
    while node:
        if node.item_id is None:
            single_sample = [item_id, node.val, 0, 1]
        else:
            single_sample = [item_id, node.item_id, 1, 1]
        samples.append(single_sample)
        positive_info[i] = node
        node = node.parent
        i += 1
    #j代表 叶子节点到root一路的index k代表当前level
    j, k = i-1, 0
    level_nodes = [root]
    while level_nodes:
        tmp = []
        for node in level_nodes:
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        if j >= 0:
            level_nodes.remove(positive_info[j])
        if level_nodes:
            if len(level_nodes) <= 2*k:
                index_list = range(len(level_nodes))
                sample_num -= len(level_nodes)
            else:
                index_list = random.sample(range(len(level_nodes)), 2*k)
                sample_num -= 2*k
            if j == 0:
                index_list = random.sample(range(len(level_nodes)), 80)
            for level_index in index_list:
                if level_nodes[level_index].item_id is None:
                    single_sample = [item_id, level_nodes[level_index].val, 0, 0]
                else:
                    single_sample = [item_id, level_nodes[level_index].item_id, 1, 0]
                samples.append(single_sample)
        level_nodes = tmp
        k += 1
        j -= 1
        if(j < 0):
            break
    o = time.clock()
    print('one batch finish %f' % (o-s))
    samples = pd.DataFrame(samples, columns=['item_ID', 'node', 'is_leaf', 'label'])
    return samples


def _tree_generate_worker(task_queue, sample_queue):
    while True:
        try:
            item_id, node ,node_list= task_queue.get()
            # ['item_ID':叶子节点id, 'node':叶子节点是val,其他是item_id, 'is_leaf', 'label']
            node_sample = _single_node_sample_1(item_id, node,node_list)
            sample_queue.put(node_sample)
        except Exception as err:
            print("Tree Worker Process Exception Info: {}".format(str(err)))
        finally:
            task_queue.task_done()


def tree_generate_samples(items, leaf_dict, node_list):
    """Sample based on the constructed tree with multiprocess."""
    jobs = mp.JoinableQueue()
    tree_samples = mp.Queue()
    for _ in range(1):
        process = mp.Process(target=_tree_generate_worker, args=(jobs, tree_samples))
        process.daemon = True
        process.start()
    total_samples = None
    for i in range(0, len(items), 1):
        sub_items = items[i:i+1]
        for item in sub_items:
            jobs.put((item, leaf_dict[item],node_list))
        #生产者调此方法进行阻塞
        jobs.join()
        batch_samples = []
        while not tree_samples.empty():
            tree_sample = tree_samples.get_nowait()
            batch_samples.append(tree_sample)
        if total_samples is None:
            total_samples = pd.concat(batch_samples, ignore_index=True)
        else:
            batch_samples = pd.concat(batch_samples, ignore_index=True)
            total_samples = pd.concat([total_samples, batch_samples], ignore_index=True)
    return total_samples


def _single_data_merge(data, tree_data):
    complete_data = None
    # tree_data ['item_ID', 'node', 'is_leaf', 'label']
    # data ['user_ID','timestamp','item','behaviors']
    item_ids = np.array(data.item_ID)
    item_ids = item_ids[item_ids != -2]
    for item in item_ids:
        samples_tree_item = tree_data[tree_data.item_ID == item][['node', 'is_leaf', 'label']].reset_index(drop=True)
        size = samples_tree_item.shape[0]
        data_extend = pd.concat([data] * size, axis=1, ignore_index=True).T
        data_item = pd.concat([data_extend, samples_tree_item], axis=1)
        if complete_data is None:
            complete_data = data_item
        else:
            complete_data = pd.concat([complete_data, data_item], axis=0, ignore_index=True)
    return complete_data


def _merge_generate_worker(tree_data, task_queue, sample_queue):
    # tree_data ['item_ID', 'node', 'is_leaf', 'label']
    # task_queue ['user_ID','timestamp','item_id','behaviors_list']
    # sample_queue:out_queue
    while True:
        try:
            data_row = task_queue.get()
            complete_sample = _single_data_merge(data_row, tree_data)
            sample_queue.put(complete_sample)
        except Exception as err:
            print("Merge Worker Process Exception Info: {}".format(str(err)))
        finally:
            task_queue.task_done()


def merge_samples(data, tree_sample):
    """combine the preprocessed samples and tree samples."""
    jobs = mp.JoinableQueue()
    complete_samples = mp.Queue()
    for _ in range(1):
        process = mp.Process(target=_merge_generate_worker, args=(tree_sample, jobs, complete_samples))
        process.daemon = True
        process.start()
    data_complete = None
    train_size = data.shape[0]
    for i in range(0, train_size, 1):
        for _ in range(1):
            if i == train_size:
                break
            jobs.put(data.iloc[i])
            i += 1
        jobs.join()
        batch_samples = []
        while not complete_samples.empty():
            single_data_sample = complete_samples.get_nowait()
            batch_samples.append(single_data_sample)
        if data_complete is None:
            data_complete = pd.concat(batch_samples, ignore_index=True)
        else:
            batch_samples = pd.concat(batch_samples, ignore_index=True)
            data_complete = pd.concat([data_complete, batch_samples], ignore_index=True)
    return data_complete


class Dataset(object):
    """construct the dataset iterator."""
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.data = self.data.drop(columns=['user_ID', 'timestamp'])
        N, B = self.data.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        if self.data.shape[1] > 2:
            return ((np.array(self.data.loc[idxs[i:i+B], 'item_ID'].tolist()),
                     self.data.loc[idxs[i:i+B], 'node'].values[:, None],
                     self.data.loc[idxs[i:i+B], 'is_leaf'].values[:, None],
                     np.array(self.data.loc[idxs[i:i+B], 'label'].tolist())) for i in range(0, N, B))
        else:
            return (np.array(self.data.loc[idxs[i:i+B], 'item_ID'].tolist()) for i in range(0, N, B))

def map_generate(df):
    r_value = {}
    df = df.values
    for i in df:
        value = r_value.get(i[0])
        if value == None:
            r_value[i[0]] = [[i[1],i[2],i[3]]]
        else:
            r_value[i[0]].append([i[1], i[2], i[3]])
            r_value[i[0]] = r_value[i[0]]
        # print('finish tree_map %f' % (o-s))
    return r_value

def merge_samples(data, tree_map):
    """combine the preprocessed samples and tree samples."""
    train_size = data.shape[0]
    r_value = []
    #[user_ID,item_ID,behavior_num] ['node', 'is_leaf', 'label']
    for i in range(train_size):
        data_row = data.iloc[i]
        data_row_values = data_row.values
        item_list = data_row.item_ID
        for item in item_list:
            l_len = len(tree_map[item])
            data_extend = pd.concat([data] * l_len, axis=1, ignore_index=True).T
            data_item = pd.concat([data_extend, tree_map[item]], axis=1)
            tmp = np.append(l_len*[data_row_values],tree_map[item],axis=1)
            print(tmp)
            r_value.extend(tmp)
        # 18917
        # print('merge finish  %f' % (o-s))
    return r_value


if __name__ == '__main__':
    # tree_data ['item_ID', 'node', 'is_leaf', 'label']
    # data ['user_ID','timestamp','item','behaviors']
    # item_ids = item_ids[item_ids != -2]
    # data_train, data_validate, data_test, cache = data_process()
    # 357722
    import numpy as np
    print(10000%10000)
    print('sdfa %s'%'a')
    t=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
    print(t.iloc[0:2,:])
    # tree_data = pd.DataFrame({'item_ID': [0,0,1,1],
    #                     'node':[1,2,3,4],
    #                     'is_leaf':[1,0,1,0],
    #                     'label': [1 for i in range(4)]})
    # # print(tree_data[tree_data['item_ID']==0])
    # data = pd.DataFrame({'user_ID': [123,456],
    #                     'item_ID':[[0],[0]],
    #                     'label': [1 for i in range(2)]})
    # # print(tree_data.iloc[0].values)
    # # print(np.append([[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]],axis=1))
    # # print(2*[[1, 2, 3], [4, 5, 6]])
    # print(tree_data.values)
    # tree_map = map_generate(tree_data)
    # print(tree_map)
    # print(tree_map)
    # merge_samples(data, tree_map)



    # user_dict, item_dict, _, tree = cache
    # tree = TreeInitialize(df)
    # root = tree.random_binary_tree()
    # items = tree.items
    # node_list = tree._node_list(root)
    # ['item_ID', 'node', 'is_leaf', 'label']
    # total_samples = tree_generate_samples(items, tree.leaf_dict, node_list)
    # print(total_samples)
    # print(total_samples)
    # print(total_samples)
    # #data_train ['user_ID', 'timestamp','item_list','behavior_type'] 加了mask的
    # #这部分代码改到spark
    # data_train = pd.DataFrame({'user_ID':100,'item_ID': [[12,2]],
    #                                    'timestamp': [6],'behaviors':[[1,1]]})
    # print(data_train)
    # pd.set_option('display.max_columns', None)
    # # user_ID  item_ID timestamp behaviors  node  is_leaf  label
    # data_complete = merge_samples(data_train, total_samples)
    # print(data_complete)
    # # item_ID , node , is_leaf , label
    # dtrain = Dataset(data_complete, 50, shuffle=True)
    # for i in dtrain:
    #     print(i)