import csv
class WordDictionary:
    def __init__(self):
        self.words={}

    def add_word(self,word):
        if not word in self.words:
            self.words[word]=len(self.words)

    def get_index(self,word):
        if word in self.words:
            return self.words[word]
        else:
            return None

    def save(self,path):
        file=open(path+"\\word_dict.csv",'w+',newline='',encoding='utf-8')
        wr=csv.writer(file)
        wr.writerows(self.words)
        file.close()

    def load(self,path):
        self.words.clear()
        file=open(path+"\\word_dict.csv",'r',encoding='utf-8')
        rd=csv.reader(file)
        for line in rd:
            strs=line.split(',')
            self.words[strs[0]]=strs[1]
        file.close()