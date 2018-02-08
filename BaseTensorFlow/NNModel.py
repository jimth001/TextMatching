# coding:utf-8
from Model.Model import Model
from Model.Config import Config
from Preprocess.Preprocessor import Preprocessor
from Preprocess.Preprocessor import WordDictionary
from datetime import timedelta
import os
import time
import tensorflow as tf
import abc
from Model.Metrics import Metrics


class NNConfig(Config):
    def __init__(self, name, task_type, metric='cross_entropy'):
        super(NNConfig, self).__init__(task_type=task_type, metric=metric)
        # 模型名称。暂时不用。
        self.nn_name = name  # nn名称,str
        # 训练及模型保存相关参数
        self.save_per_batch = 100
        self.print_per_batch = 100
        self.batch_size = 128
        self.epoch_num = 6
        '''
        self.save_per_batch = 10
        self.print_per_batch = 10
        self.batch_size = 16
        self.epoch_num = 1
        '''
        self.learning_rate = 0.0008
        self.dropout_keep_prob = 0.9
        self.require_improvement = 200 * self.print_per_batch
        # embedding相关参数
        self.embedding_dim = 256  # 词向量维度
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        self.tensorboard_dir = 'tensorboard/textrnn'


class NNModel(Model):
    def __init__(self, config):
        super(NNModel, self).__init__(config)
        self.word_dict = None
        self.target_dict = None
        self.session = None
        self.nn_inited = False
        self.inputs_data = []
        self.metrics = []
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_y = tf.placeholder(tf.float32, [None, None], name='input_r')  # 占位符不设shape，传入参数时会自行匹配

    def load(self,init_nn=True,load_dict=True):
        super(NNModel, self).load(self.config.save_dir)
        if load_dict or self.word_dict is None:
            self.word_dict = WordDictionary()
            self.word_dict.load(self.config.save_dir, 'word_dict')
        if load_dict or self.target_dict is None:
            self.target_dict = WordDictionary()
            self.target_dict.load(self.config.save_dir, 'target_dict')
        if self.nn_inited == False or init_nn:
            self._build_graph(self.word_dict.get_size())
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, self.config.save_dir)


    def feed_data(self, inputs_data, keep_prob, target=None):
        # 每个具体的model要注意定义placeholder的顺序和batch_iter返回的数据的顺序要一致，对应。
        feed_dict = {}
        for i in range(len(self.inputs_data)):
            feed_dict[self.inputs_data[i]] = inputs_data[i]
        feed_dict[self.keep_prob] = keep_prob
        if not target is None:
            feed_dict[self.input_y] = target
        return feed_dict

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

    @abc.abstractclassmethod
    def batch_iter(self, input_data, target, batch_size=64,shuffle=False):
        # 应该返回一个list对应feed_data的inputs_data,还有一个target
        return #经测试，父类写return没关系，不会导致子类调用super后函数直接结束。把它(super)当成一个普通函数来看。

    def inference_all(self, input_data):
        super(NNModel, self).inference_all(input_data)
        batches = self.batch_iter(input_data, self.config.batch_size,shuffle=False)
        y_pred = []
        for inputs in batches:
            y_batch_pred = self.session.run([self.y_pred_value], feed_dict=self.feed_data(inputs, 1.0))
            y_pred += list(y_batch_pred[0])
        return y_pred

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def _evaluate_without_predict_result(self, input_data, target,cal_mrr=False):#todo 这个函数后续必须优化。不够general
        batches = self.batch_iter(input_data, target, self.config.batch_size,shuffle=False)
        total_loss = 0.0
        total_acc = 0.0
        total_len = len(target)
        y_pred=[]
        for batch_data, batch_target in batches:
            batch_len = len(batch_target)
            feed_dict = self.feed_data(inputs_data=batch_data, keep_prob=1.0, target=batch_target)
            if cal_mrr:
                loss, acc ,y_batch_pred= self.session.run([self.loss, self.acc,self.y_pred_value], feed_dict=feed_dict)
                y_pred += list(y_batch_pred)
            else:
                loss, acc= self.session.run([self.loss, self.acc],feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        if cal_mrr:
            return total_loss / total_len, total_acc / total_len,Metrics.calculate_mrr(input_data,y_pred,target,self.target_dict)
        else:
            return total_loss / total_len, total_acc / total_len

    def _build_graph(self):  # 构建图
        print("building graph......")
        self.nn_inited = True

    def _create_placeholders(self):
        # 一个任务至少输入一个x，一个y

        pass

    def train_with_preprocessed_data(self, train_data, train_target, val_data, val_target, vocab_size, init_nn=True,
                                     release_resources=False):
        start_time = time.time()
        if init_nn:
            self._build_graph(vocab_size)  # 根据词表大小建立计算图
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
        print(str(self.__get_time_dif(start_time)) + "trainning and evaluating...")
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        flag = False  # 停止标志
        # 创建session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())  # 初始化全局变量
        for epoch in range(self.config.epoch_num):
            print("epoch:" + str(epoch + 1))
            batch_train = self.batch_iter(train_data, train_target, batch_size=self.config.batch_size,shuffle=True)
            for batch_data, batch_target in batch_train:
                feed_dict = self.feed_data(inputs_data=batch_data, target=batch_target,
                                           keep_prob=self.config.dropout_keep_prob)
                s = self.session.run([self.optim, merged_summary],
                                     feed_dict=feed_dict)
                if total_batch==0:#初次存一下。因为测试代码时数据很少，可能到不了要保存的batch数就结束了。
                    saver.save(sess=self.session, save_path=self.config.save_dir)  # 保存当前的训练结果
                if total_batch % self.config.save_per_batch == 0:  # 每多少轮次将训练结果写入tensorboard scalar
                    writer.add_summary(s[1], total_batch)  # 传入summary信息和当前的batch数
                if total_batch > 0 and total_batch % self.config.print_per_batch == 0:  # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.keep_prob] = 1.0  # 在验证集上验证时dropout的概率改为0
                    # 算一下在这个train_batch上的loss和acc
                    loss_train, acc_train = self.session.run([self.loss, self.acc],
                                                             feed_dict=feed_dict)
                    loss_val, acc_val = self._evaluate_without_predict_result(val_data, val_target)  # 验证，得到loss和acc
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
                if total_batch - last_improved > self.config.require_improvement:
                    # 早停：验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出batch
            if flag:  # 跳出epoch
                break
        if release_resources:
            self.session.close()
            self.session = None

    def train_onehotkey(self, train_path, init_nn=True, release_resources=False, weight_balanced=False,data_preprocessed=False):  # 一键式训练
        super(NNModel, self).train_onehotkey(train_path)
        if data_preprocessed==False:
            word_dict, target_dict = Preprocessor.preprocess_and_save(
                train_path, self.config.save_dir,
                task_type=self.config.task_type, weight_balanced=weight_balanced)
        train_data,train_target,val_data,val_target,vocab_size=Preprocessor.load_preprocessed_data_for_train(self.config.save_dir,task_type=self.config.task_type)
        self.train_with_preprocessed_data(train_data,train_target,val_data,val_target,vocab_size,init_nn=init_nn,release_resources=release_resources)

    def inference_onehotkey(self, test_path,init_nn=True,load_model=True,data_preprocessed=False):
        #inference,does not care the metrics on test data set,only return the result model predicted
        if data_preprocessed==False:
            self.word_dict=WordDictionary()
            self.target_dict=WordDictionary()
            self.word_dict.load(self.config.save_dir,"word_dict")
            self.target_dict.load(self.config.save_dir,"target_dict")
            self.word_dict,self.target_dict= Preprocessor.preprocess_and_save(test_path,
                                            self.config.save_dir, task_type=self.config.task_type, name="test",
                                            weight_balanced=False, word_dict=self.word_dict,
                                            target_dict=self.target_dict)
        if load_model==False:
            self.load(init_nn=init_nn,load_dict=False)
        x_test,y_test=Preprocessor.load_preprocessed_data_for_test(self.config.save_dir,task_type=self.config.task_type)
        return self.inference_all(x_test)

    def evaluate_onehotkey(self, test_path, init_nn=True,load_model=True,data_preprocessed=False):
        if data_preprocessed == False:
            self.word_dict = WordDictionary()
            self.target_dict = WordDictionary()
            self.word_dict.load(self.config.save_dir, "word_dict")
            self.target_dict.load(self.config.save_dir, "target_dict")
            self.word_dict, self.target_dict = Preprocessor.preprocess_and_save(test_path,
                                                                                self.config.save_dir,
                                                                                task_type=self.config.task_type,
                                                                                name="test",
                                                                                weight_balanced=False,
                                                                                word_dict=self.word_dict,
                                                                                target_dict=self.target_dict)
        x_test, y_test = Preprocessor.load_preprocessed_data_for_test(self.config.save_dir,
                                                                      task_type=self.config.task_type)
        if len(y_test)==0:
            raise ValueError("test data has no label，cannot evaluate!")
        if load_model:#模型没有加载。因此要加载。
            self.load(init_nn=init_nn,load_dict=False)
        print(self._evaluate_without_predict_result(x_test,y_test,cal_mrr=True))
        #result=self.inference_all(x_test)