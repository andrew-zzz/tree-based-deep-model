import tensorflow as tf
import os
from .sample_init import *
# from Dice import dice
import sklearn.metrics as metrics

class Model(object):

    def __init__(self, album_count, node_count):
        self.album_count = album_count
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]    #node_id
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]  #label
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B] 播放历史个数
        self.lr = tf.placeholder(tf.float64, [])      #decay
        self.is_leaf = tf.placeholder(tf.int32, [None, ]) #node节点0 叶子节点1
        hidden_units = 32
        self.saver = None

        self.item_emb_w = tf.get_variable("item_emb_w", [album_count, hidden_units])
        item_b = tf.get_variable("item_b", [album_count],
                                 initializer=tf.constant_initializer(0.0))

        node_emb_w = tf.get_variable("node_emb_w", [node_count, hidden_units])
        node_b = tf.get_variable("node_b", [node_count],
                                 initializer=tf.constant_initializer(0.0))

        #处理item_embedding
        i_emb = tf.nn.embedding_lookup(self.item_emb_w,self.i) #[B,H]
        n_emb = tf.nn.embedding_lookup(node_emb_w,self.i) #[B,H]
        i_b = tf.gather(item_b, self.i)
        n_b = tf.gather(node_b, self.i)

        # Mask 根据leaf的值取出album还是node的 embdding,bias组装
        key_masks = tf.expand_dims(self.is_leaf,-1)  # [B, 1]
        key_masks = tf.tile(key_masks, [1,hidden_units]) #[B,H]
        key_masks = key_masks > 0
        i_emb = tf.where(key_masks, i_emb, n_emb)  # [B, H]

        key_masks_1 = self.is_leaf > 0
        i_b = tf.where(key_masks_1, i_b, n_b) #[B]

        #历史embedding and attention
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist_i)
        hist_i = attention(i_emb, h_emb, self.sl)

        # -- attention end ---
        # hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

        u_emb_i = hist_i

        # -- fcn begin -------
        din_i = tf.concat([u_emb_i, i_emb], axis=-1)
        # din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 64, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 32, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        self.logits = i_b + d_layer_3_i
        self.predictions = tf.sigmoid(self.logits)
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_i = tf.reshape(self.score_i, [-1, 1])

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            list(zip(clip_gradients, trainable_params)), global_step=self.global_step)

    def train(self, sess, train_set, l):
        #uid 0
        #play_list 1-5
        #list_len  6
        #node_id   7
        #is_leaf   8
        #label     9
        # (i, y, is_leaf, hist_i, sl)
        # self.i = tf.placeholder(tf.int32, [None, ])  # [B]    #node_id
        # self.y = tf.placeholder(tf.float32, [None, ])  # [B]  #label
        # self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        # self.sl = tf.placeholder(tf.int32, [None, ])  # [B] 播放历史个数
        # self.lr = tf.placeholder(tf.float64, [])      #decay
        # self.is_leaf = tf.placeholder(tf.int32, [None, ]) #node节点0 叶子节点1
        # print('train')
        features,label = sess.run(train_set)
        features = np.array(features)
        # print('features')
        # exit(0)
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.i: features[:,7],
            self.y: label,
            self.is_leaf: features[:,8],
            self.hist_i: features[:,1:6],
            self.sl: features[:,6],
            self.lr: l
        })
        return loss

    def _eval(self,sess,model,test_set,validation_step):
        score_arr = []
        p_arr = []
        for i in range(validation_step):
        # for _, uij in DataInput(test_set, test_batch_size):
            features, label = sess.run(test_set)
            features = np.array(features)
            score = sess.run(model.predictions, feed_dict={
                self.i: features[:, 7],
                self.is_leaf: features[:,8],
                self.hist_i: features[:,1:6],
                self.sl: features[:, 6],
            })
            score_arr.extend(score)
            p_arr.extend(label)
        test_auc = metrics.roc_auc_score(p_arr, score_arr)
            # 保存pb
            # from tensorflow.python.framework import graph_util
            # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #                                                            ['in_i', 'in_hist', 'in_sl', 'prediction'])
            # with tf.gfile.FastGFile('/home/dev/data/andrew.zhu/vip/din/model/pb/buy_model.pb',
            #                         mode='wb') as f:  # 模型的名字是model.pb
            #     f.write(constant_graph.SerializeToString())
        return test_auc

    def predict(self,data,sess):
        val = sess.run(self.predictions, feed_dict={
            self.i: data[:, 7],
            self.is_leaf: data[:, 8],
            self.hist_i: data[:, 1:6],
            self.sl: data[:, 6],
        })
        return val

    def get_embeddings(self,item_list,save_path):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            item_embeddings = sess.run(tf.nn.embedding_lookup(self.item_emb_w, np.array(item_list)))
            # print(item_embeddings.tolist())
            return item_embeddings.tolist()

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    '''
      queries:     [B, H]
      keys:        [B, T, H]
      keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs