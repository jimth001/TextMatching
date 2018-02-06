# coding:utf-8
from .WordDictionary import WordDictionary
import numpy as np
from .Tokenizer import Tokenizer
import codecs
import csv


class Preprocessor:
    @staticmethod
    def load_data(data_path, task_type="matching"):  # 加载数据。不检查格式。
        if task_type == "classification":  # 分类任务，要求数据格式为：文本 标签（中间使用\t分隔）
            x_data = []
            y_data = []
            has_y = True
            file = codecs.open(data_path, 'r', encoding='utf-8')
            for line in file:
                strs = line.strip('\r\n').strip('\n').split('\t')
                x_data.append(strs[0])
                if has_y:
                    try:
                        y_data.append(strs[1])  # 注意target是str
                    except IndexError:
                        y_data = None
                        has_y = False
            return x_data, y_data
        elif task_type == "matching":  # 文本匹配任务，要求的数据格式为：query response 标签(中间使用\t分隔)
            q_data = []  # query
            r_data = []  # response
            y_data = []
            has_y = True
            file = codecs.open(data_path, 'r', encoding='utf-8')
            for line in file:
                strs = line.strip('\r\n').strip('\n').split('\t')
                q_data.append(strs[0])
                r_data.append(strs[1])
                if has_y:
                    try:
                        y_data.append(strs[2])  # 注意target是str
                    except IndexError:
                        y_data = None
                        has_y = False
            return q_data, r_data, y_data

    @staticmethod
    def target_2_one_hot(y_data,
                         target_dict=None):
        # 对target编码并向量化。适用于multi-class classification。允许target是各种形式，比如“类别1，类别2”
        is_new_dict = False
        if target_dict == None:
            target_dict = WordDictionary()
            is_new_dict = True
        y_index = []
        for tar in y_data:
            index = target_dict.get_index(tar)
            if index is None:
                if is_new_dict:
                    target_dict.add_word(tar)
                    index = target_dict.get_index(tar)
                else:
                    raise ValueError("未知的类别：" + str(tar))
            y_index.append(index)
        size = target_dict.get_size()
        y_new = []
        for index in y_index:
            tmp = [0] * size
            tmp[index] = 1
            y_new.append(tmp)
        return y_new, target_dict

    @staticmethod
    def seg_and_2_int(x_data, word_dict=None, dict_append=False):
        # 若word_dict不为空。且dict_append为True，则会更新词典。
        tokenizer = Tokenizer()
        new_x = []
        # todo 增加统计词频
        word_frequency_dict = {}
        if word_dict is None:
            dict = WordDictionary()
            for sen in x_data:
                sen_vec = []
                seg_rst = tokenizer.parser(sen).split(' ')
                for word in seg_rst:
                    index = dict.get_index(word)
                    if index is None:
                        dict.add_word(word)
                        sen_vec.append(dict.get_index(word))
                    else:
                        sen_vec.append(index)
                new_x.append(sen_vec)
            word_dict = dict
        else:
            for sen in x_data:
                sen_vec = []
                seg_rst = tokenizer.parser(sen)
                for word in seg_rst:
                    index = word_dict.get_index(word)
                    if index is None:  # 如果用已有词典，且dict_append=False,那么出现未登录词的时候丢弃这个词
                        if dict_append:
                            word_dict.add_word(word)
                            sen_vec.append(word_dict.get_index(word))
                    else:
                        sen_vec.append(index)
                new_x.append(sen_vec)
        return new_x, word_dict

    @staticmethod
    def save_preprocessed_data(data, path, name,task_type="matching"):
        # data：要存储的数据
        # path：存储路径
        # name：文件名称
        if task_type=="matching":
            file = codecs.open(path + name, 'w+', encoding='utf-8')
            wr = csv.writer(file)
            for d in data:
                wr.writerow(d[0])
                wr.writerow(d[1])
            file.close()

    @staticmethod
    def load_preprocessed_data(data_path, name,task_type="matching"):
        # data_path文件存储路径
        # name文件存储名称

        # todo

        pass

    @staticmethod
    def preprocess_and_save(data_path, store_path, task_type="matching", name="train", weight_balanced=False,
                                       word_dict=None,
                                       target_dict=None):
        #for both train and test
        # 一些通用的预处理操作：
        # 载入训练集,划分验证集：
        if name=="train":
            dict_append=True
        else:
            dict_append=False
        if task_type == "matching":
            print("loading data......")
            q_train, r_train, y_train = Preprocessor.load_data(data_path, task_type)
            print("converting sentence 2 word-index vector......")
            q_train, word_dict = Preprocessor.seg_and_2_int(x_data=q_train, word_dict=word_dict, dict_append=dict_append)
            r_train, word_dict = Preprocessor.seg_and_2_int(x_data=r_train, word_dict=word_dict, dict_append=dict_append)
            if weight_balanced and len(y_train) > 0:
                print("balancing data......")
                q_r_train = [[q_train[x], r_train[x]] for x in range(len(q_train))]  # packed q and r
                q_r_train = Preprocessor.get_balanced_data(q_r_train, y_train)
            if len(y_train) > 0:  # 是有label的样本
                print("converting target 2 one-hot vector......")
                y_train, target_dict = Preprocessor.target_2_one_hot(y_train, target_dict=target_dict)
                target_dict.save(store_path, 'target_dict')
                Preprocessor.save_preprocessed_data(y_train, store_path, 'y_' + name)
            print("saving......")
            word_dict.save(store_path, 'word_dict')
            Preprocessor.save_preprocessed_data(q_r_train, store_path, 'fea_' + name)
            return word_dict, target_dict

    @staticmethod
    def get_balanced_data(data, target, rate=0.95, random_flow_rate=0.1):  # 过采样得到均衡数据集
        # 训练样本才需要平衡，因此target不能为空
        # rate from 0 to 1,表示少数样本过采样后与多数样本的基准比例
        # random_flow_rate 表示rate的浮动值
        # rate+rand(random_flow_rate)=少数样本/多数样本
        # todo 加入对rate和random_flow_rate的支持。加入对rate和random_flow_rate范围的合法性检查。
        if len(data) != len(target):
            raise ValueError("x和y长度不一致")
        target_num_counter = {}
        target_data_dict = {}
        new_data = [x for x in data]
        new_target = [x for x in target]
        for i in range(len(target)):
            if target[i] in target_num_counter:
                target_num_counter[target[i]] += 1
            else:
                target_num_counter[target[i]] = 1
            if target[i] in target_data_dict:
                target_data_dict[target[i]].append(i)
            else:
                target_data_dict[target[i]] = [i]
        max = 0
        for key in target_num_counter.keys():
            if target_num_counter[key] > max:
                max = target_num_counter[key]
        for key in target_data_dict.keys():
            list = target_data_dict[key]
            length = len(list)
            for i in range(max - length):
                random_index = np.random.randint(0, length)
                new_data.append(data[list[random_index]])
                new_target.append(target[list[random_index]])
        return new_data, new_target

    @staticmethod
    def batch_iter(x, y, batch_size=64, padding=0):  # 生成批次数据。x中数据可以不等长，会padding成一样的
        # todo 加入对y=None(即推断数据)的支持
        # todo 适应NNModel的修改 2018-2-6
        if y is None or x is None:
            raise ValueError("x or y is None")
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = [x[i] for i in indices]
        y_shuffle = [y[i] for i in indices]
        mask = [len(x_shuffle[i]) for i in range(len(x_shuffle))]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            rx = x_shuffle[start_id:end_id]
            ry = y_shuffle[start_id:end_id]
            rm = mask[start_id:end_id]
            max_len = max(rm)
            for list in rx:
                if len(list) < max_len:
                    list += [padding] * (max_len - len(list))
            yield rx, ry, rm

    @staticmethod
    def generate_train_and_cross_validation(x, y, n_fold=4):  # 根据x，y划分n-fold交叉验证的数据集
        if len(x) != len(y):
            raise ValueError("x和y长度不一致")
        val_len = int(len(x) / n_fold)
        start_id = 0
        indices = np.random.permutation(np.arange(len(x)))
        x_shuffle = [x[i] for i in indices]
        y_shuffle = [y[i] for i in indices]
        for i in range(n_fold):
            x_train = [x_shuffle[r] for r in range(len(x)) if (r < start_id or r > start_id + val_len)]
            y_train = [y_shuffle[r] for r in range(len(x)) if (r < start_id or r > start_id + val_len)]
            x_val = [x_shuffle[r] for r in range(len(x)) if (r >= start_id and r <= start_id + val_len)]
            y_val = [y_shuffle[r] for r in range(len(x)) if (r >= start_id and r <= start_id + val_len)]
            start_id = start_id + val_len + 1
            yield x_train, y_train, x_val, y_val
