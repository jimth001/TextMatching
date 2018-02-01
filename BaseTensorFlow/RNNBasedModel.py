from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from Model import Config
from Model import Model
from Preprocess.Preprocessor import Preprocessor
from Preprocess.WordDictionary import WordDictionary
import os
from datetime import timedelta
import time



class RNNConfig(Config.Config):
    def __init__(self, config_path=None):
        super(RNNConfig, self).__init__(task_type=None,metric=None)
        if config_path is None:
            self.batch_size = 100
            self.classes_num = 2
            self.embedding_dim = 500  # 词向量维度
            self.mlp_hidden_layers_num = 128  # mlp的隐含层神经元个数
            self.hidden_dim = 2 * self.embedding_dim  # 隐藏层神经元。RNN输出的维数。如果要做多层RNN的话，输入和输出维数必须相同。
            # 因此，这里隐藏神经元数量是embedding_dim的2倍
            self.rnn = 'lstm'  # rnn类型
            self.vocab_size = 5000  # 词表长度
            self.layers_num = 2  # RNN层数
            self.learning_rate = 0.1
            self.epoch_num = 10000
            self.dropout_keep_prob = 0.9
            self.train_path = ""
            self.test_path = ""


class RNNModel(Model.Model):
    def __init__(self, config):  # config是配置信息
        super(RNNModel, self).__init__(config)
        self.config = config
        #self.batch_seq_len=tf.placeholder(tf.int32,name='batch_seq_len')
        self.input_x = tf.placeholder(tf.int32, name='input_x')  # placeholder只存储一个batch的数据
        self.input_y = tf.placeholder(tf.float32, name='input_y')#占位符不设shape，传入参数时会自行匹配
        self.mask = tf.placeholder(tf.int32,[self.config.batch_size], name='mask')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.word_dict = None
        self.rnn()  # 这个是干啥的？

    # 输入数据是用在词典中的id表示单词的，因此要有一个统一的词典
    def rnn(self):
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
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)  # embedding_inputs是embedding的结果

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for i in range(self.config.layers_num)]
            rnn_cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32,
                                            sequence_length=self.mask)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数：交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)  # 对tensor所有元素求平均
            # 优化器。指定优化方法，学习率，最大化还是最小化，优化目标函数
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def feed_data(self, x_batch, y_batch, mask, keep_prob):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.mask: mask,
            self.keep_prob: keep_prob
        }
        return feed_dict

    def evaluate(self, sess, x_, y_):  # 这个应该在基类中实现
        """评估在某一数据上的准确率和损失"""
        # todo config对根据不同任务不同指标进行评价
        data_len = len(x_)
        batch_eval = Preprocessor.batch_iter(x_, y_, 128)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / data_len, total_acc / data_len

    def get_time_dif(start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def train(self, train_path):
        super(RNNModel, self).train(train_path)
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        tensorboard_dir = 'tensorboard/textrnn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        # 这里应该是定义网络的summary信息包括loss和accuracy，并进行merge。以后run一个batch指定fetch的东西为merged_summary就可以得到loss和accuracy
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        # 配置 Saver
        saver = tf.train.Saver()
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        ##############################################################
        print("trainning and evaluating...")
        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 1000
        flag = False
        # 载入训练集,划分验证集
        x_train, y_train = Preprocessor.load_data(self.config.train_path)
        x_train, dict = Preprocessor.seg_and_2_int(x_data=x_train)
        data= Preprocessor.generate_train_and_cross_validation(x=x_train, y=y_train)
        x_train, y_train, x_val, y_val =data[0]
        dict.save("dict")
        self.word_dict=dict
        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        for epoch in range(self.config.epoch_num):
            print("epoch:" + str(epoch + 1))
            batch_train = Preprocessor.batch_iter(x_train, y_train, batch_size=self.config.batch_size)
            for x_batch, y_batch, mask in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, mask, self.config.dropout_keep_prob)
                if total_batch % self.config.save_per_batch == 0:  # 每多少轮次将训练结果写入tensorboard scalar
                    s = session.run(merged_summary, feed_dict=feed_dict)  # merged_summary定义取什么结果返回
                    writer.add_summary(s, total_batch)  # 传入summary信息和当前的batch数
                if total_batch % self.config.print_per_batch == 0:  # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.keep_prob] = 1.0  # 在验证集上验证时dropout的概率改为0
                    loss_train, acc_train = session.run([self.loss, self.acc],
                                                        feed_dict=feed_dict)  # 算一下在这个train_batch上的loss和acc
                    loss_val, acc_val = self.evaluate(session, x_val, y_val)  # 验证，得到loss和acc
                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=self.config.save_path)  # 保存当前的训练结果
                        improved_str = '*'
                    else:
                        improved_str = ''
                    time_dif = self.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(self.optim, feed_dict=feed_dict)  # 运行优化。第一个参数是run时要处理的对象（是优化器时，在计算完毕后更新参数，否则只是计算）
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 早停：验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出batch
            if flag:  # 跳出epoch
                break

    def inference_one(self, x_test):  # x_test为分好词的list,没有padding
        super(RNNModel, self).inference_one(x_test)
        # content = unicode(message)
        data = [self.word_dict[x] for x in x_test if x in self.word_dict]
        feed_dict = {
            self.input_x: data,
            self.keep_prob: 1.0
        }
        y_pred_cls = self.session.run(self.y_pred_cls, feed_dict=feed_dict)
        #return self.categories[y_pred_cls[0]]
        return y_pred_cls

    def inference_batch(self, x_test):
        max_len = len(x_test[0])  # 求batch的最大长度
        for i in range(len(x_test)):
            if max_len < len(x_test[i]):
                max_len = len(x_test[i])
        #todo
