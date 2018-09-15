from optimizer import SGD as optm
from Nueron import NeuralNetwork
import numpy as np

# generate data

# use Xiao Xue's rgb transform data in this Demo
data = np.load("train.npy")
label = np.load("label.npy")

# print (len(data),len(label))
# print (data[0],label[0])


# train
epochs = 100
epoch_content = 20000
batch_size = 64
lr = 0.01
momentum = 0.9

layer_nums = [8,16,16,8]
num_layers = len(layer_nums)
layer_types = ['input'] + ['FC' for i in range(num_layers+1)]
input_dim = output_dim = 3
Net = NeuralNetwork(batch_size=batch_size,learning_rate=lr,num_layers=num_layers,layer_types=layer_types,layer_nums=layer_nums,input_dim=input_dim,output_dim=output_dim,momentum=momentum)

opt = optm(batch_size)

for ec in range(epochs):
    print ("Epoch ",ec,":")
    for ba in range(int(epoch_content/batch_size)) :
        batch_data,batch_label = opt.select_input(data=data,label=label)
        Net.init_update_layers()# init layers
        loss_batch = 0.0
        for iter in range(batch_size):
            loss_batch += Net.forward(batch_data[iter],batch_label[iter])
            Net.backward(Net.Layers[-1].net, Net.Layers[-1].out, batch_label[iter])

        # update together
        Net.update_layers()

        if ba % 40 == 0:
            loss_batch = loss_batch / float(batch_size)
            print ("[batch",ba,"] : ","loss: ",loss_batch)

    print ("\n")

# for debug
# Demo in csdn : https://www.cnblogs.com/charlotte77/p/5629865.html#4057715

# # train
# epochs = 100
# epoch_content = 20000
# batch_size = 1
# lr = 0.5
# num_layers = 1
#
# layer_types = ['insert'] + ['FC' for i in range(num_layers+1)]
# layer_nums = [2]
# input_dim = output_dim = 2
# Net = NeuralNetwork(batch_size=batch_size,learning_rate=lr,num_layers=num_layers,layer_types=layer_types,layer_nums=layer_nums,input_dim=input_dim,output_dim=output_dim)
#
# opt = optm(batch_size)

# Net.init_update_layers()# init layers
#
# input = np.array([0.05,0.1])
# result = np.array([0.01,0.99])
# weights0=np.array([[0.15,0.25],[0.20,0.30]])
# weights1=np.array([[0.40,0.50],[0.45,0.55]])
# Net.Layers[1].set_weights(weights0)
# Net.Layers[1].set_bias(0.35)
# Net.Layers[2].set_weights(weights1)
# Net.Layers[2].set_bias(0.60)

# # forward没问题
#loss = Net.forward(input,result)

# print ("input:")
# print (Net.Layers[0].out)
# print ("1 weights")
# print (Net.Layers[1].weights)
# print (Net.Layers[1].bias)
# print ("hidden")
# print (Net.Layers[1].net)
# print (Net.Layers[1].out)
# print ("2 weights")
# print (Net.Layers[2].weights)
# print (Net.Layers[2].bias)
# print ("output")
# print (Net.Layers[2].net)
# print (Net.Layers[2].out)
# print ("result")
# print (result)
# print (loss)

# # bakcward 没问题
# Net.init_update_layers()
# Net.backward(Net.Layers[-1].net,Net.Layers[-1].out,result)
# print ("change")
# print (Net.Layers[-1].update_weights)
# print (Net.Layers[-1].update_bias)
# print ("before update:")
# print (Net.Layers[-1].weights)
# print (Net.Layers[-1].bias)
# Net.update_layers()
# print ("after update:")
# print (Net.Layers[-1].weights)
# print (Net.Layers[-1].bias)
# print ("hidden weights an bias")
# print (Net.Layers[1].weights)
# print (Net.Layers[1].bias)