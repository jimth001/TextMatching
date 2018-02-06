# coding:utf-8
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.contrib import rnn
from BaseTensorFlow.NNModel import NNModel, NNConfig


class RNNConfig(NNConfig):
    def __init__(self, name,config_path=None):  ##这里是网络独有的结构的相关参数
        super(RNNConfig, self).__init__(name,task_type=None, metric=None)
        if config_path is None:
            self.mlp_hidden_layers_num = 128  # mlp的隐含层神经元个数
            self.hidden_dim = self.embedding_dim  # 隐藏层神经元。
            self.rnn = 'gru'  # rnn类型
            self.layers_num = 2  # RNN层数


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
        self.inputs_data(self.q_sequence_len)
        self.inputs_data(self.r_sequence_len)


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
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)  # embedding_inputs是embedding的结果

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for i in range(self.config.layers_num)]
            rnn_cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
            #todo 不同rnn cell输入输出是什么，该怎么取返回值
            _outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32,
                                            sequence_length=self.sequence_len)# 取state作为结果


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(state[0], self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc2')
            self.y_pred_class = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数：交叉熵
            #tf.nn.sparse_softmax_cross_entropy_with_logits()#这个可以不用把y变成one-hot的向量
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)  # 对tensor所有元素求平均
            # 优化器。指定优化方法，学习率，最大化还是最小化，优化目标函数
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_class)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def batch_iter(self, input_data, target, batch_size=64):
        #该模型的数据结构为[[q_list],[r_list]]
        #[[q,r]*n]
        super(RNNModel,self).batch_iter(input_data, target, batch_size=64)
        #todo:


