import math
import numpy as np

def relu(net):
    out = []
    for i in range(len(net)):
        out[i].append(max(0,net[i]))
    return np.asarray(out)

class sigmoid():
    def forward(self,net):
        out = []
        for i in range(len(net)):
            cur = 1.0 / (1.0 + math.exp(-net[i]))
            out[i].append(cur)
        return np.asarray(out)

    def backward(self,net,out):
        assert len(net) == len(out), "net 和 out 长度不符"
        grad = np.zeros(len(net))
        for i in range(len(net)):
            grad[i] = out[i] * (1-out[i])

        return grad
