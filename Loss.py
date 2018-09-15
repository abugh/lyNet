import numpy as np
from Activation import nothing

class L2_loss:
    def __init__(self):
        pass
    def forward(self,output,label):
        sum_loss = 0.0
        assert len(output)==len(label),"output 和 label 长度不符"
        for i in range(len(output)):
            sum_loss += (label[i]-output[i]) ** 2
        return sum_loss / 2.0

    def backward(self,Layers,net,output,label,actv=nothing(),lr=0.05):
        assert len(output) == len(label), "output 和 label 长度不符"
        grad = actv.backward(net, output)# grad between net and out (for activate)
        alpha_nets = np.zeros(len(output))
        for i in range(len(output)):
            ahpha_out = (output[i]-label[i])
            alpha_net = ahpha_out * grad[i]
            alpha_nets[i] = alpha_net
            # calculate update_weights and update_bias
            for j in range(len(Layers[-1].weights)):# P.s. len(Layers[-1].weights) == layer_nums[-2]
                Layers[-1].update_weights[j][i] += lr * alpha_net * Layers[-2].out[j]
            Layers[-1].update_bias += lr * alpha_net
        return alpha_nets
