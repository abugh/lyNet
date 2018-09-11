import numpy as np
import random
from Layer import FullyConnectedLayer
from Activation import sigmoid as activate
from Loss import L2_loss as loss_func
from optimizer import SGD as optm

class NeuralNetwork :
    def __init__(self,optimizer='SGD',batch_size=4,learning_rate = 0.5,num_layers=5,layer_types=None,layer_nums=None,input_dim = None,output_dim=None):
        self.opt = optimizer
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_layers = num_layers# todo:这是隐层的数量，不包括input和output
        assert (layer_types is not None) and (layer_nums is not None) and (input_dim is not None) and (output_dim is not None),"参数不能为None"
        assert (len(layer_types-1)==len(layer_nums)==num_layers),"隐层参数的个数不符"# 输出层和最后一层隐层之间也需要指定连接方式
        # 把输入输出层也加进去,这样for循环能好写一点
        self.all_layer_nums = layer_nums
        self.all_layer_nums.insert(0,input_dim)
        self.all_layer_nums.append(output_dim)


        # 初始化层
        self.Layers = []
        for i in range(0,num_layers+2):# 输入输出层和隐层都需要初始化结点存储空间
            if layer_types[i] == 'FC':  # Fully Connected
                if i == 0:
                    self.Layers.append(FullyConnectedLayer(0,input_dim,0))
                else:
                    self.Layers.append(FullyConnectedLayer(i,self.all_layer_nums[i], self.all_layer_nums[i-1]))

    # todo : save和load check_point
    def save_check_point(self):
        return

    def save_check_point(self,check_point):
        return

    def forward(self,input,label):
        self.Layers[0].set_input(input)
        for i in range(1,self.num_layers+2) :
            for j in range(self.all_layer_nums[i]):
                net_neron = 0
                for k in range(self.all_layer_nums[i-1]):
                    net_neron += self.Layers[i].weights[k][j] * self.Layers[i-1].out[k]
                self.Layers[i].net[j] = net_neron
            if i != 0:  # input 层不要activation
                out = activate.forward(self.Layers[i].net)
            else:
                out = self.Layers[i].net.copy()
            self.Layers[i].out = out
        return loss_func.forward(self.Layers[-1].out,label)


    def backward(self,label):
        # update output layer's weights,using loss backward
        last_alpha_net = loss_func.backward( self.Layers,label,activate,self.lr)
        for i in reversed(range(1,self.num_layers+1)):# output层的update# 在loss_func的backward中已经做好了
            alpha_out = np.zeros(self.all_layer_nums[i])
            # calculate gradient for hidden layer
            for j in range(self.all_layer_nums[i]):
                for k in range(self.all_layer_nums[i+1]):# 从后面的layer累加gradient
                    alpha_out[j] += last_alpha_net[k] * self.Layers[i+1].weights[j][k]
            grad = activate.backward(self.Layers[i].net, self.Layers[i].out)
            alpha_net = alpha_out * grad
            # update hidden layer weights
            for j in range(self.all_layer_nums[i]):
                for k in range(self.all_layer_nums[i-1]):
                    self.Layers[i].weights[k][j] -= self.lr * alpha_net[j] * self.Layers[i-1].out[k]

            last_alpha_net = alpha_net
