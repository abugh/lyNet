import random
import numpy as np

class Layer:
    def __init__(self,type,id):
        self.type = type
        self.id = id
    def forward(self):# 动态图版本会用到,目前不打算做动态图的
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    def __init__(self, type='FN',id=None,channels=64,last_channels=None):
        '''
        :param type: layer的类型
        :param channels: 通道数
        :param weights: 用于load或者初始化权重
        '''
        Layer.__init__(self,type,id)
        self.net = np.assarray([0.0 for i in range(channels)])
        self.out = np.assarray([0.0 for i in range(channels)])
        assert last_channels is not None,"没有输入上一层的channels数"
        assert id is not None, "没有输入id"
        if last_channels == 0:# input layer
            self.weights = None
        else:
            self.weights = np.assarray([[random.random() for h1 in range(channels)] for h2 in range(last_channels)])

    def set_input(self,input):
        assert self.id==0, "设置了非input层"
        self.net = input.copy()
        self.out = input.copy()
