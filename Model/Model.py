from sklearn.metrics.classification import classification_report
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics.regression import r2_score

class Model:
    def __init__(self,config):
        self.is_debug=True
        self.config=config

    def train(self,x_train,y_train):
        if self.is_debug:
            print("train:------------------------")
        pass


    def test(self,x_test,y_test=None):
        if not y_test is None:
            if len(x_test)!=len(y_test):
                raise ValueError("x_test和y_test长度不一致")
            return self.inference(x_test)
        else:
            result=self.inference(x_test)
            self.evaluate(result,y_test)

    def evaluate(self,y_predict,y_true,target_names=None):
        if self.config.task_type=='classification':
            classification_report(y_true=y_true,y_pred=y_predict,target_names=target_names)
        elif self.config.task_type=='ranking':
            roc_auc_score(y_true=y_true,y_score=y_predict)
        elif self.config.task_type=='regression':
            r2_score(y_true=y_true,y_pred=y_predict)

    def inference_one(self,x_test):
        if self.is_debug:
            print("inference:")
        return None

    def cross_val(self,data_path):
        if self.is_debug:
            print("inference:"+data_path)
        pass

    def train_with_config(self):
        if self.is_debug:
            print("train_with_config")
        pass

    def inference_with_config(self):
        if self.is_debug:
            print("inference_with_config")
        pass

    def save(self):
        pass

    def load(self):
        pass