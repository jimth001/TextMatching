from BaseTensorFlow.RNNBasedModel import RNNModel,RNNConfig

config=RNNConfig('my_rnn')
config.rnn='gru'
rnn=RNNModel(config)
rnn._build_graph(vocab_size=300)
print("finished")