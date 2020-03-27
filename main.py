#! D:\E\PROFESSIONAL\Anaconda3\envs\pytorch\python.exe
#MLP测试学习程序
import numpy as np
import torch as th
import torchvision as thv
import mlp_classes
import time
import matplotlib.pyplot as plt

print('test')
print("GPU可用数目：",th.cuda.device_count())

train_set = thv.datasets.MNIST(
    r"D:\E\Project\GitRepository\MLP_MNIST\MNIST\train",
    train=True,
    transform=thv.transforms.ToTensor(),
    download=True)
test_set = thv.datasets.MNIST(
    r"D:\E\Project\GitRepository\MLP_MNIST\MNIST\test",
    train=False,
    transform=thv.transforms.ToTensor(),
    download=True)
#将数据整理成batch格式，并转换成可迭代对象
train_dataset = th.utils.data.DataLoader(train_set,100)
test_dataset = th.utils.data.DataLoader(test_set,100)


model = mlp_classes.MLP().cuda()
print(model)

# 损失函数与优化器
optimizer = th.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
lossfunc = th.nn.CrossEntropyLoss().cuda()

# test accuarcy
#print(mlp_classes.AccuarcyCompute(
#    np.array([[1,10,6],[0,2,5]],dtype=np.float32),
#    np.array([[1,2,8],[1,2,5]],dtype=np.float32)))

#训练：
for x in range(4):
    for i,data in enumerate(train_dataset):
        optimizer.zero_grad()
    
        (inputs,labels) = data
        inputs = th.autograd.Variable(inputs).cuda()
        labels = th.autograd.Variable(labels).cuda()
    
        outputs = model(inputs)
    
        loss = lossfunc(outputs,labels)
        loss.backward()
    
        optimizer.step()
    
        if i % 100 == 0:
            print(i,":",mlp_classes.AccuarcyCompute(outputs,labels))

#测试：
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = th.autograd.Variable(inputs).cuda()
    labels = th.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(mlp_classes.AccuarcyCompute(outputs,labels))
print("准确率:",sum(accuarcy_list) / len(accuarcy_list))













