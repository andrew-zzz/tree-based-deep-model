import pickle
from ximalaya_brain_utils.hdfs_util import HdfsClient

def _node_list(root):
# 将二叉树数据提出放入list
    def node_val(node):
        if (node.left or node.right):
            return str(node.val)+'-'+ str(0)
        else:
            return str(node.item_id)+'-'+ str(1)

    node_queue = [root]
    arr_arr_node = []
    arr_arr_node.extend([node_val(node_queue[0])])
    while node_queue:
        tmp = []
        tmp_val = []
        for i in node_queue:
            if i is None:
                tmp.append(None)
                tmp.append(None)
                tmp_val.append("")
                tmp_val.append("")
            else:
                if i.left:
                    tmp.append(i.left)
                    tmp_val.append(node_val(i.left))
                else:
                    tmp.append(None)
                    tmp_val.append("")
                if i.right:
                    tmp.append(i.right)
                    tmp_val.append(node_val(i.right))
                else:
                    tmp.append(None)
                    tmp_val.append("")
        node_queue = tmp
        is_break = True
        for j in tmp:
            if j != None:
                is_break = False
        if is_break:
            break
        else:
            arr_arr_node.extend(tmp_val)
    return arr_arr_node

def print_last_layer(root):
# 将二叉树数据提出放入list
    node_queue = [root]
    arr_arr_node = []
    layer = 0
    total = 0
    while node_queue:
        tmp = []
        for i in node_queue:
            if i is None:
               continue
            else:
                if i.left:
                    tmp.append(i.left)
                if i.right:
                    tmp.append(i.right)
        node_queue = tmp
        print_out = []
        for i in node_queue:
            if(i.item_id is not None):
                print_out.append((i.item_id,1))
            else:
                print_out.append((i.val,0))
        print('layer %d' % layer)
        total = total + len(print_out)
        layer = layer + 1
    return arr_arr_node


def main():
    hdfs_client = HdfsClient()
    with open('/home/dev/data/andrew.zhu/tdm/data_flow/final_tree.pkl', 'rb') as f:
        tree = pickle.load(f)
        # print_last_layer(tree.root)
        with open('/home/dev/data/andrew.zhu/tdm/data_flow/tree_str', "w", encoding='utf-8') as f:
            out = str(_node_list(tree.root))
            f.write(out)
            f.close()
        hdfs_client.upload("/user/dev/andrew.zhu/tdm/model/tree_str.txt", "/home/dev/data/andrew.zhu/tdm/data_flow/tree_str", overwrite=True)

    with open('/home/dev/data/andrew.zhu/tdm/data_flow/sample.pkl', 'rb') as f:
        data_train = pickle.load(f)
        data_validate = pickle.load(f)
        user_dict, item_index, tree = pickle.load(f)

        with open('/home/dev/data/andrew.zhu/tdm/data_flow/item_dict', "w", encoding='utf-8') as f:
            f.write(str(item_index))
            f.close()

        hdfs_client.upload("/user/dev/andrew.zhu/tdm/model/item_dict.txt", "/home/dev/data/andrew.zhu/tdm/data_flow/item_dict", overwrite=True)
