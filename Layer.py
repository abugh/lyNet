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
    def __init__(self, type='FC',id=None,channels=64,last_channels=None,momentum=0.9):
        '''
        :param type: layer的类型
        :param channels: 通道数
        :param weights: 用于load或者初始化权重
        '''
        self.channels = channels
        self.last_channels = last_channels
        self.momentum = momentum
        Layer.__init__(self,type,id)
        self.net = np.asarray([0.0 for i in range(channels)])
        self.out = np.asarray([0.0 for i in range(channels)])
        assert last_channels is not None,"没有输入上一层的channels数"
        assert id is not None, "没有输入id"
        if last_channels == 0:# input layer
            self.weights = None
            self.update_weights = None
            self.weights_v = None# for momentums
            self.bias = None
            self.update_bias = None
            self.bias_v = None# for momentums
        else:
            self.weights = np.asarray([[random.uniform(0,1) for h1 in range(channels)] for h2 in range(last_channels)],dtype=np.float32)
            self.update_weights = np.asarray([[0.0 for h1 in range(channels)] for h2 in range(last_channels)],dtype=np.float32)
            self.weights_v = np.asarray([[0.0 for h1 in range(channels)] for h2 in range(last_channels)],dtype=np.float32)
            self.bias = 0
            self.update_bias = 0
            self.bias_v = 0



    def set_input(self,input):
        assert self.id==0, "设置了非input层"
        self.net = input.copy()
        self.out = input.copy()

    # init update_weights and update_bias with 0
    def init_update(self):
        self.weights_v = np.asarray([[0.0 for h1 in range(self.channels)] for h2 in range(self.last_channels)],dtype=np.float32)
        self.bias_v = 0
        self.update_weights = np.asarray([[0.0 for h1 in range(self.channels)] for h2 in range(self.last_channels)],dtype=np.float32)
        self.update_bias = 0

    # update weights and bias
    def update(self):
        # print (self.id,"  更新量")
        # print (self.update_weights)
        # print (self.update_bias)
        # print ("")
        self.weights_v = self.momentum * self.weights_v - self.update_weights
        self.bias_v = self.momentum * self.bias_v - self.update_bias
        self.weights += self.weights_v
        self.bias += self.bias_v

    def set_weights(self,weights):# 用于debug和load_check_point
        self.weights = weights.copy()

    def set_bias(self,bias):# 用于debug和load_check_point
        self.bias = bias