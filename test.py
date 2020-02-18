from construct_tree import TreeInitialize

def preorder(self, root):
    if root is None:
        return ''
    if root.lef:
        self.preorder(root.left)
    if root.right:
        self.preorder(root.right)

def leaf(root,list):
    if root == None:
        return 0
    elif root.left == None and root.right == None:
        list.append(root.item_id)
        return list
    else:
        leaf(root.left, list)
        leaf(root.right, list)
    return list

def map_generate(df):
    #生成map 为了提高访问速度
    r_value = {}
    df = df.values
    for i in df:
        value = r_value.get(i[0])
        if value == None:
            r_value[i[0]] = [[i[1],i[2],i[3]]]
        else:
            r_value[i[0]].append([i[1], i[2], i[3]])
            r_value[i[0]] = r_value[i[0]]
    return r_value

def _node_list1(root):
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


def printTree(root):
    if not root:
        return
    print('Binary Tree:')
    printInOrder(root, 0, 'H', 17)


def printInOrder(root, height, preStr, length):
    if not root:
        return
    printInOrder(root.right, height + 1, 'v', length)
    string = preStr + str(root.val) + preStr
    leftLen = (length - len(string)) // 2
    rightLen = length - len(string) - leftLen
    res = " " * leftLen + string + " " * rightLen
    print(" " * height * length + res)
    printInOrder(root.left, height + 1, '^', length)



def _node_list(root):
# 将二叉树数据提出放入list
    def node_val(node):
        if (node.left or node.right):
            return (node.val, 0)
        else:
            return (node.item_id, 1)

    node_queue = [root]
    arr_arr_node = []
    arr_arr_node.append([node_val(node_queue[0])])
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
            arr_arr_node.append(tmp_val)
    return arr_arr_node


if __name__ == '__main__':
    a = []
    import pandas as pd
    import numpy as np
    data = pd.DataFrame({'item_ID':range(20),'category_ID':range(20)})
    # data1= data.sample(frac=1).reset_index(drop=True)
    # print(data1)
    tree = TreeInitialize(data)
    tree.random_binary_tree()
    print(leaf(tree.root,a))
    r = _node_list(tree.root)
    r1 = _node_list1(tree.root)
    print(r)
    print(r1)
    #
    #
    # import numpy as np
    # a = np.array([[1,2,3],[4,4,5]])
    # print(a.shape)