import random

class optimizer:
    def __init__(self,batch_size):
        self.batch_size = batch_size
    def select_input(self,data):
        raise NotImplementedError

class SGD(optimizer):
    def __init__(self,batch_size=1):
        optimizer.__init__(self,batch_size)
    def select_input(self,data):
        samples_list = range(len(data))
        indxis = random.sample(samples_list,self.batch_size)

        input = []
        for i in range(self.batch_size):
            input.append(data[indxis[i]])

        return input