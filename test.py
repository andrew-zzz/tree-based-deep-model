from construct_tree import TreeInitialize

def preorder(self, root):
    if root is None:
        return ''
    if root.lef:
        self.preorder(root.left)
    if root.right:
        self.preorder(root.right)

def leaf(root,list):
    if root==None:
        return 0
    elif root.left==None and root.right==None:
        list.append(root.item_id)
        return list
    else:
        leaf(root.left,list)
        leaf(root.right,list)
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

if __name__ == '__main__':
    a = [1,2,3,4]
    import pandas as pd
    import numpy as np
    data = pd.DataFrame({'item_ID':[1,1,2,5,1],'category_ID':range(5),'category_ID1':range(5),'category_ID2':range(5)})
    print(np.array(data.values)[:,1:3])

    # tree = TreeInitialize(data)
    # tree.random_binary_tree()
    # a = []
    # leaf(tree.root,a)
    # print(a)
    #
    #
    # import numpy as np
    # a = np.array([[1,2,3],[4,4,5]])
    # print(a.shape)