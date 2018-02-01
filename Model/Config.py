class Config:#config文件错误的话应该抛异常停止
    def __init__(self,config_path=None):
        pass

    def __init__(self,task_type,metric):
        self.task_type=task_type
        self.metric=metric

    def Load(self,config_path):
        pass

    def Save(self,config_path):
        pass