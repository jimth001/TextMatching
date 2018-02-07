# coding:utf-8
from sklearn.metrics.classification import classification_report
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics.regression import r2_score
import abc


class Model(object):
    @abc.abstractclassmethod
    def __init__(self, config, is_debug=True):
        self.is_debug = is_debug
        self.config = config

    @abc.abstractclassmethod
    def train_onehotkey(self, train_path):  # 一键深度学习
        #训练过程中保存模型
        if self.is_debug:
            print("train:-------------------------------------------------")

    @abc.abstractclassmethod
    def inference_onehotkey(self, ):
        pass

    @abc.abstractclassmethod
    def evaluate_onehotkey(self, y_predict, y_true, target_names=None):
        pass
        '''if self.config.task_type=='classification':
            classification_report(y_true=y_true,y_pred=y_predict,target_names=target_names)
        elif self.config.task_type=='ranking':
            roc_auc_score(y_true=y_true,y_score=y_predict)
        elif self.config.task_type=='regression':
            r2_score(y_true=y_true,y_pred=y_predict)'''

    @abc.abstractclassmethod
    def inference_one(self, x_test):
        if self.is_debug:
            print("inference_one:---------------------------------------------")

    @abc.abstractclassmethod
    def inference_all(self, x_test):
        if self.is_debug:
            print("inference_all:---------------------------------------------")

    @abc.abstractclassmethod
    def load(self, path):
        pass
