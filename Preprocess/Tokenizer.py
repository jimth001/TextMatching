import thulac

class Tokenizer:
    def __init__(self):
        self.user_dict=None
        self.model_path=None#默认为model_path
        self.T2S=True#繁简体转换
        self.seg_only=True#只进行分词
        self.filt=False#去停用词
        self.tokenizer=thulac.thulac(user_dict=self.user_dict,model_path=self.model_path,T2S=self.T2S,seg_only=self.seg_only,filt=self.filt)

    def parser(self,text):
        return self.tokenizer.cut(text,text=True)#返回文本

tokenizer=Tokenizer()
print(tokenizer.parser("设置用户词典，用户词典中的词会被打上uw标签。词典中每一个词一行，UTF8编码"))