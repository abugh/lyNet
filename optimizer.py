import random

class optimizer:
    def __init__(self,batch_size):
        self.batch_size = batch_size
    def select_input(self,data,label):
        raise NotImplementedError

class SGD(optimizer):
    def __init__(self,batch_size=1):
        optimizer.__init__(self,batch_size)
    def select_input(self,data,label):
        assert len(data)==len(label),"data和label长度不等"
        samples_list = range(len(data))
        indxis = random.sample(samples_list,self.batch_size)

        input_data = []
        input_label = []
        for i in range(self.batch_size):
            input_data.append(data[indxis[i]])
            input_label.append(label[indxis[i]])
        return input_data,input_label