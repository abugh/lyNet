import numpy as np
from Layer import FullyConnectedLayer
from Activation import sigmoid as activate
from Activation import nothing as output_activate
# from Activation import sigmoid as output_activate
from Loss import L2_loss as loss_func


class NeuralNetwork :
    def __init__(self,optimizer='SGD',batch_size=16,learning_rate = 0.5,num_layers=5,layer_types=None,layer_nums=None,input_dim = None,output_dim=None,momentum=0.9):
        self.momentum = momentum
        self.opt = optimizer
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_layers = num_layers# todo:这是隐层的数量，不包括input和output
        assert (layer_types is not None) and (layer_nums is not None) and (input_dim is not None) and (output_dim is not None),"参数不能为None"
        assert (len(layer_types)-2==len(layer_nums)==num_layers),"隐层参数的个数不符"# 输出层和最后一层隐层之间也需要指定连接方式
        # 把输入输出层也加进去,这样for循环能好写一点
        self.all_layer_nums = layer_nums
        self.all_layer_nums.insert(0,input_dim)
        self.all_layer_nums.append(output_dim)


        # 初始化层
        self.Layers = []
        for i in range(0,num_layers+2):# 输入输出层和隐层都需要初始化结点存储空间
            if i == 0:
                self.Layers.append(FullyConnectedLayer(id=0, channels=input_dim, last_channels=0, momentum=self.momentum))
            elif layer_types[i] == 'FC':  # Fully Connected
                self.Layers.append(FullyConnectedLayer(id=i,channels=self.all_layer_nums[i], last_channels=self.all_layer_nums[i-1], momentum=self.momentum))

    # todo : save和load check_point
    def save_check_point(self):
        return

    def load_check_point(self,check_point):
        return

    def forward(self,input,label):
        self.Layers[0].set_input(input)
        for i in range(1,self.num_layers+2) :
            for j in range(self.all_layer_nums[i]):
                net_neron = 0
                for k in range(self.all_layer_nums[i-1]):
                    net_neron += self.Layers[i].weights[k][j] * self.Layers[i-1].out[k]
                net_neron += self.Layers[i].bias
                self.Layers[i].net[j] = net_neron
            if i != 0 and i != self.num_layers+1 :  # input 和 output 层不要activation
                out = activate().forward(self.Layers[i].net)
            elif i == 0:
                out = self.Layers[i].net.copy()
            elif i == self.num_layers+1:# output层专用的activation
                out = output_activate().forward(self.Layers[i].net)
            self.Layers[i].out = out
        return loss_func().forward(self.Layers[-1].out,label)


    def backward(self,net,output,label):
        # update output layer's weights,using loss backward
        last_alpha_net = loss_func().backward( Layers=self.Layers,net=net,output=output,label=label,actv=output_activate(),lr=self.lr)
        for i in reversed(range(1,self.num_layers+1)):# output层的update在loss_func的backward中已经做好了
            alpha_out = np.zeros(self.all_layer_nums[i])
            # calculate gradient for hidden layer
            for j in range(self.all_layer_nums[i]):
                for k in range(self.all_layer_nums[i+1]):# 从后面的layer累加gradient
                    alpha_out[j] += last_alpha_net[k] * self.Layers[i+1].weights[j][k]
            grad = activate().backward(self.Layers[i].net, self.Layers[i].out)
            alpha_net = alpha_out * grad
            #  calculate hidden layers' update_weights and update_bias
            for j in range(self.all_layer_nums[i]):
                for k in range(self.all_layer_nums[i-1]):
                    self.Layers[i].update_weights[k][j] += self.lr * alpha_net[j] * self.Layers[i-1].out[k]
                self.Layers[i].update_bias += self.lr * alpha_net[j]

            last_alpha_net = alpha_net

    def init_update_layers(self):
        for i in range(1,self.num_layers+2):# input层没有weights 和 bias
            self.Layers[i].init_update()

    # 最后再一起update整个batch
    def update_layers(self):
        for i in range(1,self.num_layers+2):
            self.Layers[i].update()

