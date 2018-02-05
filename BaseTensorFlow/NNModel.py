from Model.Model import Model
from Model.Config import Config
from Preprocess.Preprocessor import Preprocessor
from Preprocess.Preprocessor import WordDictionary
from datetime import timedelta
import os
import time
import tensorflow as tf


class NNConfig(Config):
    def __init__(self, name, task_type='classification', metric='cross_entropy'):
        super(NNConfig, self).__init__(task_type=task_type, metric=metric)
        # 模型名称。暂时不用。
        self.nn_name = name  # nn名称,str
        # 训练及模型保存相关参数
        self.save_per_batch = 50
        self.print_per_batch = 50
        self.batch_size = 50
        self.learning_rate = 0.001
        self.epoch_num = 2000
        self.dropout_keep_prob = 0.9
        # embedding相关参数
        self.embedding_dim = 500  # 词向量维度
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        self.tensorboard_dir = 'tensorboard/textrnn'


class NNModel(Model):
    def __init__(self, config):
        super(NNModel, self).__init__(config)
        self.word_dict = None
        self.target_dict = None
        self.session = None
        self.nn_inited = False

    def load(self):
        super(NNModel, self).load(self.config.save_dir)
        self.word_dict = WordDictionary()
        self.word_dict.load(self.config.save_dir, 'word_dict')
        self.target_dict = WordDictionary()
        self.target_dict.load(self.config.save_dir, 'target_dict')
        if self.nn_inited == False:
            self._build_graph(self.word_dict.get_size())
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, self.config.save_dir)

    def save(self):
        super(NNModel, self).save(self.config.save_dir)
        self.word_dict.save(self.config.save_dir, 'word_dict')
        self.target_dict.save(self.config.save_dir, 'target_dict')
        saver = tf.train.Saver()
        saver.save(sess=self.session, save_path=self.config.save_dir)  # 保存当前的训练结果
        self.session.close()
        self.session = None

    def inference_one(self, x_test):  # x_test为分好词的list,没有padding
        super(NNModel, self).inference_one(x_test)
        # content = unicode(message)
        data = [self.word_dict[x] for x in x_test if x in self.word_dict]
        feed_dict = {
            self.input_x: data,
            self.keep_prob: 1.0
        }
        y_pred_class = self.session.run(self.y_pred_class, feed_dict=feed_dict)
        # return self.categories[y_pred_class[0]]
        return y_pred_class

    def inference_all(self, x_):
        super(NNModel, self).inference_all(x_)
        y_ = [0] * len(x_)
        batches = Preprocessor.batch_iter(x_, y_, self.config.batch_size)
        y_pred = []
        for x_batch, y_batch, sequence_len_batch in batches:
            feed_dict = {
                self.input_x: x_batch,
                self.sequence_len: sequence_len_batch,
                self.keep_prob: 1.0
            }
            y_batch_pred = self.session.run([self.y_pred_class], feed_dict=feed_dict)
            y_pred += list(y_batch_pred[0])
        return y_pred

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def _evaluate_without_predict_result(self, x_, y_):
        batches = Preprocessor.batch_iter(x_, y_, self.config.batch_size)
        total_loss = 0.0
        total_acc = 0.0
        total_len = len(x_)
        for x_batch, y_batch, sequence_len_batch in batches:
            batch_len = len(y_batch)
            feed_dict = self.feed_data(x_batch, y_batch, sequence_len_batch, keep_prob=1.0)
            loss, acc = self.session.run([self.loss, self.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / total_len, total_acc / total_len

    def _build_graph(self):  # 构建图
        print("building graph......")
        self.nn_inited = True

    def _create_placeholders(self):
        pass

    def train(self, train_path, init_nn=True, save_and_quit=False, weight_balanced=False):
        super(NNModel, self).train(train_path)
        # 载入训练集,划分验证集：
        x_train, y_train = Preprocessor.load_data(train_path)
        if weight_balanced:
            x_train, y_train = Preprocessor.get_balanced_data(x_train, y_train)
        x_train, self.word_dict = Preprocessor.seg_and_2_int(x_data=x_train)
        y_train, self.target_dict = Preprocessor.target_2_one_hot(y_train)
        self.word_dict.save("model_store\\dict", "word_dict")
        self.target_dict.save("model_store\\dict", "target_dict")
        data = Preprocessor.generate_train_and_cross_validation(x=x_train, y=y_train, n_fold=4)
        x_train, y_train, x_val, y_val = data.__next__()
        # 初始化网络结构。nn结构有依赖train_data的参数。因为读完train_data才知道词表大小，要根据词表大小确定embedding层的size。
        if init_nn:
            self._build_graph(self.word_dict.get_size())
        if not os.path.exists(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        # 这里应该是定义网络的summary信息包括loss和accuracy，并进行merge。以后run一个batch指定fetch的东西为merged_summary就可以得到loss和accuracy
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.config.tensorboard_dir)
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
        flag = False  # 停止标志
        # 创建session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())  # 初始化全局变量
        for epoch in range(self.config.epoch_num):
            print("epoch:" + str(epoch + 1))
            batch_train = Preprocessor.batch_iter(x_train, y_train, batch_size=self.config.batch_size)
            for x_batch, y_batch, sequence_len_batch in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, sequence_len_batch, self.config.dropout_keep_prob)
                s = self.session.run([self.optim, merged_summary], feed_dict=feed_dict)
                if total_batch % self.config.save_per_batch == 0:  # 每多少轮次将训练结果写入tensorboard scalar
                    writer.add_summary(s[1], total_batch)  # 传入summary信息和当前的batch数
                if total_batch % self.config.print_per_batch == 0:  # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.keep_prob] = 1.0  # 在验证集上验证时dropout的概率改为0
                    loss_train, acc_train = self.session.run([self.loss, self.acc],
                                                             feed_dict=feed_dict)  # 算一下在这个train_batch上的loss和acc
                    loss_val, acc_val = self.__evaluate_without_predict_result(x_val, y_val)  # 验证，得到loss和acc
                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=self.session, save_path=self.config.save_dir)  # 保存当前的训练结果
                        improved_str = '*'
                    else:
                        improved_str = ''
                    time_dif = self.__get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                total_batch += 1
                if total_batch - last_improved > require_improvement:
                    # 早停：验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出batch
            if flag:  # 跳出epoch
                break
        if save_and_quit:
            self.save()
