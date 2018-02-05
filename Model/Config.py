# coding:utf-8
class Config:  # config文件错误的话应该抛异常停止
    def __init__(self, task_type, metric):
        self.task_type = task_type
        self.metric = metric
        self.classes_num = 2
        self.save_dir = "model_store/"

    def load(self, config_path):
        pass

    def save(self, config_path):
        pass
