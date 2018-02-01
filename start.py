from BaseTensorFlow import RNNBasedModel
import os
config=RNNBasedModel.RNNConfig()
config.Load(os.getcwd()+"\config")
rnn=RNNBasedModel.RNNModel(config)
rnn.train("E:\\Learning\\研一\\文本匹配\\NLPCC2017-OpenDomainQA\\training&testing\\nlpcc-iccpol-2016.dbqa.training-data")
rnn.inference_with_config()