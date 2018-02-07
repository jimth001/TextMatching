# coding:utf-8
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.contrib import rnn
from BaseTensorFlow.NNModel import NNModel, NNConfig
import numpy as np


class RNNConfig(NNConfig):
    def __init__(self, name,config_path=None):  ##这里是网络独有的结构的相关参数
        super(RNNConfig, self).__init__(name,task_type=None, metric=None)
        if config_path is None:
            self.mlp_hidden_layers_num = 128  # mlp的隐含层神经元个数
            self.hidden_dim = self.embedding_dim  # rnn_cell隐藏层神经元个数。
            self.rnn = 'lstm'  # rnn类型。可以选lstm和gru
            self.layers_num = 1  # RNN层数

class RNNModel(NNModel):
    def __init__(self, config):  # config是配置信息
        super(RNNModel, self).__init__(config)
        self.config = config
        self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
        self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
        self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
        self.r_sequence_len = tf.placeholder(tf.int32, [None],name='r_sequence_len')
        self.inputs_data.append(self.input_q)
        self.inputs_data.append(self.input_r)
        self.inputs_data.append(self.q_sequence_len)
        self.inputs_data.append(self.r_sequence_len)

    def build_graph_for_test(self,vocab_size=3000):
        self._build_graph(vocab_size=vocab_size)
    # 输入数据是用在词典中的id表示单词的，因此要有一个统一的词典

    def _build_graph(self, vocab_size):
        super(RNNModel,self)._build_graph()
        def lstm_cell():
            return rnn.BasicLSTMCell(self.config.hidden_dim)

        def gru_cell():
            return rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):  # 指定cpu0执行
            embedding = tf.get_variable('embedding', [vocab_size, self.config.embedding_dim])
            query_embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_q)  # embedding_inputs是embedding的结果
            response_embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_r)

        with tf.name_scope("rnn"):
            #query和response使用同一个RNN
            cell_for_query = [dropout() for i in range(self.config.layers_num)]# 多层rnn网络
            rnn_cell_for_query = rnn.MultiRNNCell(cell_for_query, state_is_tuple=True)
            query_outputs, query_state = tf.nn.dynamic_rnn(cell=rnn_cell_for_query, inputs=query_embedding_inputs, dtype=tf.float32,
                                            sequence_length=self.q_sequence_len)# 取state作为结果
            response_outputs, response_state = tf.nn.dynamic_rnn(cell=rnn_cell_for_query, inputs=response_embedding_inputs, dtype=tf.float32,
                                                sequence_length=self.r_sequence_len)  # 取state作为结果
            '''sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
            Used to copy-through state and zero-out outputs when past a batch
            element's sequence length.  So it's more for correctness than performance.'''

        with tf.name_scope("rank"):
            matrix=tf.get_variable('matrix',[self.config.hidden_dim,self.config.hidden_dim],trainable=True)
            #todo
            # 全连接层，后面接dropout以及relu激活
            if self.config.rnn=='lstm':#return (c,h),h is output,c is the hidden state
                # todo q*mat*r得到一个数，拼到特征向量中
                interaction=tf.matmul(a=tf.matmul(query_state[0][1],matrix),b=response_state[0][1],transpose_b=True)
                fc_input = tf.concat([query_state[0][1],interaction, response_state[0][1]], axis=1)  # todo 搞清这个得好好看看原理和读源码。到底该取0还是1
            else:#GRU只有一个state
                interaction = tf.matmul(a=tf.matmul(query_state[0], matrix), b=response_state[0],transpose_b=True)
                fc_input=tf.concat([query_state[0],interaction,response_state[0]])
                # todo q*mat*r得到一个数，拼到特征向量中
            fc = tf.layers.dense(fc_input, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc2')
            self.y_pred_value = tf.nn.softmax(self.logits)  # 输出概率值(相似度值)
            self.y_pred_class = tf.argmax(self.y_pred_value, 1)# 预测类别

        with tf.name_scope("optimize"):
            # 损失函数：交叉熵
            #tf.nn.sparse_softmax_cross_entropy_with_logits()#这个可以不用把y变成one-hot的向量
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)  # 对tensor所有元素求平均
            # 优化器。指定优化方法，学习率，最大化还是最小化，优化目标函数为交叉熵
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("evaluate metrics"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_class)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def batch_iter(self, input_data, target=None, batch_size=64,padding=0):
        #该模型的数据结构为[[q_list],[r_list]]
        super(RNNModel,self).batch_iter(input_data, target, batch_size=64)
        if len(input_data[0])!=len(input_data[1]):
            raise ValueError("input_data:q and r are not in same length")
        if input_data is None:
            raise ValueError("input_data is None")
        data_len = len(input_data[0])
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        q_shuffle = [input_data[0][i] for i in indices]
        r_shuffle = [input_data[1][i] for i in indices]
        q_seq_len = [len(q_shuffle[i]) for i in range(len(q_shuffle))]
        r_seq_len = [len(r_shuffle[i]) for i in range(len(r_shuffle))]
        if target is None:
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_q = q_shuffle[start_id:end_id]
                batch_r = r_shuffle[start_id:end_id]
                batch_q_seq_len = q_seq_len[start_id:end_id]
                batch_r_seq_len = r_seq_len[start_id:end_id]
                q_max_len = max(batch_q_seq_len)
                r_max_len = max(batch_r_seq_len)
                for list in batch_q:
                    if len(list) < q_max_len:
                        list += [padding] * (q_max_len - len(list))
                for list in batch_r:
                    if len(list)<r_max_len:
                        list+=[padding]*(r_max_len-len(list))
                yield [batch_q,batch_r,batch_q_seq_len,batch_r_seq_len]
        else:
            y_shuffle = [target[i] for i in indices]
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_q = q_shuffle[start_id:end_id]
                batch_r = r_shuffle[start_id:end_id]
                batch_y = y_shuffle[start_id:end_id]
                batch_q_seq_len = q_seq_len[start_id:end_id]
                batch_r_seq_len = r_seq_len[start_id:end_id]
                q_max_len = max(batch_q_seq_len)
                r_max_len = max(batch_r_seq_len)
                for list in batch_q:
                    if len(list) < q_max_len:
                        list += [padding] * (q_max_len - len(list))
                for list in batch_r:
                    if len(list)<r_max_len:
                        list+=[padding]*(r_max_len-len(list))
                yield [batch_q,batch_r,batch_q_seq_len,batch_r_seq_len],batch_y


