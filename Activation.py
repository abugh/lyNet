import math
import numpy as np

class nothing:
    def __init__(self):
        pass

    def forward(self, net):
        return net.copy()

    def backward(self, net, out):
        return np.ones(len(net))

class sigmoid:
    def __init__(self):
        pass
    def forward(self,net):
        out = []
        for i in range(len(net)):
            cut_off_net = net[i] if net[i]>-100 else -float("inf")# 超过-100就当作-inf处理，不然exp时候会报错
            cur = 1.0 / (1.0 + math.exp(-cut_off_net))
            out.append(cur)
        return np.asarray(out)

    def backward(self,net,out):
        assert len(net) == len(out), "net 和 out 长度不符"
        grad = np.zeros(len(net))
        for i in range(len(net)):
            grad[i] = out[i] * (1-out[i])

        return grad
