import torch as th
import torchvision as thv
import numpy as np

class MLP(th.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = th.nn.Linear(784,512)
        self.fc2 = th.nn.Linear(512,128)
        self.fc3 = th.nn.Linear(128,10)

    def forward(self,din):
        din = din.view(-1,28*28)
        dout = th.nn.functional.relu(self.fc1(din))
        dout = th.nn.functional.relu(self.fc2(dout))
        return th.nn.functional.softmax(self.fc3(dout))

# accuracy 准确性
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)








