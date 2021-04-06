# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:52:47 2021

@author: 17845
"""


import DRN_DWWC_code

from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import numpy as np
import torch
from visdom import Visdom
import torch.nn as nn
import torch
import torch.nn.functional as F
#parameters
epochs = 2
batch_size = 100
lr = 0.01
#处理数据
x_train = np.load('x_train.npy'); y_train = np.load('y_train.npy')
x_valid = np.load('x_valid.npy'); y_valid = np.load('y_valid.npy')
x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train)
x_valid, y_valid = torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid)
train_data = torch.utils.data.TensorDataset(x_train, y_train)
val_data = torch.utils.data.TensorDataset(x_valid, y_valid)
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
###########################################################################################################
#DRN_DWWC
drn_dwwc = DRN_DWWC(num_classes=9, m=4) 
'''
#可视化网络结构
import netron    
import torch.onnx 
drn_dwwc = DRN_DWWC(num_classes=9) 
onnx_path = "drn_dwwc_net.onnx"
x = torch.randn(1,1,64,64)
torch.onnx.export(drn_dwwc, x, onnx_path)
netron.start(onnx_path)
'''
opimizer = optim.Adam(drn_dwwc.parameters(), lr=lr, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

viz = Visdom()
viz.line([[0.0, 0.0, 0.0]], [0.], win='loss', opts=dict(title='loss', legend=['train_loss', 'val_loss', 'val_acc']))
global_step = 0
###########################################################
# evalute model
def evalute(model, loader):
    model.eval()
    correct = 0
    test_loss = 0
    total = len(loader.dataset)
    criteon = nn.CrossEntropyLoss()
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            loss = criteon(logits, y.long())
        correct += torch.eq(pred, y).sum().float().item()
        test_loss += loss.item()*y.size()[0]
    return correct/total, test_loss/total
####################################################
#train and test
for epoch in range(epochs):
    for step,(x,y) in enumerate(train_loader):
        drn_dwwc.train()
        b_x = Variable(x)
        b_y = Variable(y)
        
        out = drn_dwwc(b_x)
        loss = loss_func(out, b_y.long())
        opimizer.zero_grad()
        loss.backward()
        opimizer.step()
        
        global_step += 1
        val_acc, val_loss = evalute(drn_dwwc, val_loader)
        viz.line([[loss.item(), val_loss, val_acc]], [global_step], win='loss', update='append')
     
        if step % 50 == 0:
             print('Epoch:',epoch,'|train loss:'+str(loss.item()))
           
#保存训练好的模型和重新加载模型
torch.save(drn_dwwc.state_dict(), 'drn_dwwc.pt')

m_state_dict = torch.load('drn_dwwc.pt')
new_m = DRN_DWWC(num_classes=9) 
new_m.load_state_dict(m_state_dict)

###############################################################################
# k折交叉验证
###############################################################################
########k折划分############        
def get_k_fold_data(k, i, X, y): 
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    index = list(range(0, y.shape[0]))
    idx = list(range(i * fold_size, (i + 1) * fold_size))
    tridx = list(set(index) - set(idx))
    x_train = X[tridx]; y_train = y[tridx]
    x_valid = X[idx]; y_valid = y[idx]

    return x_train, y_train, x_valid, y_valid
#train and test
X = np.load('x_data.npy');Y = np.load('y_data.npy')
X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y)
k = 5

train_loss_sum, valid_loss_sum = 0, 0
train_acc_sum, valid_acc_sum = 0, 0
for i in range(k):
    x_train, y_train, x_valid, y_valid = get_k_fold_data(k, i, X, Y)
    
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_valid, y_valid)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    
    drn_dwwc = DRN_DWWC(num_classes=9, m=2) 
    opimizer = optim.Adam(drn_dwwc.parameters(), lr=lr, weight_decay=0.001)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            drn_dwwc.train()
            b_x = Variable(x)
            b_y = Variable(y)
    
            out = drn_dwwc(b_x)
            loss = loss_func(out, b_y.long())
            opimizer.zero_grad()
            loss.backward()
            opimizer.step()
    
            if step % 20 == 0:
                print('Epoch:',epoch,'|train loss:'+str(loss.item()))
           
    train_loss_sum += loss.item()
    val_acc, val_loss = evalute(drn_dwwc, val_loader)
    valid_acc_sum += val_acc
    valid_loss_sum += val_loss
print('train_loss'+str(train_loss_sum/k)+'valid_loss'+str(valid_loss_sum/k)+'valid_acc'+str(valid_acc_sum/k))
  

########################################################################################
#与其他模型对比
########################################################################################    
#DRN
class FirstBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=2):
        super(FirstBlock, self).__init__()
        #卷积分支
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__() 
        #卷积分支
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #残差分支
        self.shortcut = nn.Sequential()

    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class AdjustBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=2):

        super(AdjustBlock, self).__init__() 
        #卷积分支
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
 
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #残差分支
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
                )
    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class DRN(nn.Module):

    def __init__(self, num_classes=10, m=2):   
        super(DRN, self).__init__()
        self.layer1 = FirstBlock(in_planes=1, planes=m)
        self.layer2 = AdjustBlock(in_planes=m, planes=m, stride=2)
        self.layer3 = BasicBlock(in_planes=m, planes=m, stride=1)
        self.layer4 = BasicBlock(in_planes=m, planes=m, stride=1)
        self.layer5 = AdjustBlock(in_planes=m, planes=m*2, stride=2)
        self.layer6 = BasicBlock(in_planes=2*m, planes=2*m, stride=1)
        self.layer7 = BasicBlock(in_planes=2*m, planes=2*m,  stride=1)
        self.layer8 = AdjustBlock(in_planes=2*m, planes=4*m, stride=2)
        self.layer9 = BasicBlock(in_planes=4*m, planes=4*m,  stride=1)
        self.layer10 = BasicBlock(in_planes=4*m, planes=4*m,  stride=1)
        self.bn = nn.BatchNorm2d(4*m)
        
        self.linear = nn.Linear(4*m*1*1, num_classes)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
drn = DRN(num_classes=9, m=2)
opimizer = optim.Adam(drn.parameters(), lr=lr, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

viz = Visdom()
viz.line([[0.0, 0.0, 0.0]], [0.], win='loss', opts=dict(title='loss', legend=['train_loss', 'val_loss', 'val_acc']))
global_step = 0
#train and test
for epoch in range(epochs):
    for step,(x,y) in enumerate(train_loader):
        drn.train()
        b_x = Variable(x)
        b_y = Variable(y)

        out = drn(b_x)
        loss = loss_func(out, b_y.long())
        opimizer.zero_grad()
        loss.backward()
        opimizer.step()

        global_step += 1
        val_acc, val_loss = evalute(drn, val_loader)
        viz.line([[loss.item(), val_loss, val_acc]], [global_step], win='loss', update='append')
        
        if step % 20 == 0:
            print('Epoch:',epoch,'|train loss:'+str(loss.item()))

###############################################################################################################################
#CNN
class FirstBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=2):
        super(FirstBlock, self).__init__()
        #卷积分支
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__() 
        #卷积分支
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))

        return out

class AdjustBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=2):

        super(AdjustBlock, self).__init__() 
        #卷积分支
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
 
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))

        return out

class CNN(nn.Module):

    def __init__(self, num_classes=10, m=2):   
        super(DRN, self).__init__()
        self.layer1 = FirstBlock(in_planes=1, planes=m)
        self.layer2 = AdjustBlock(in_planes=m, planes=m, stride=2)
        self.layer3 = BasicBlock(in_planes=m, planes=m, stride=1)
        self.layer4 = BasicBlock(in_planes=m, planes=m, stride=1)
        self.layer5 = AdjustBlock(in_planes=m, planes=m*2, stride=2)
        self.layer6 = BasicBlock(in_planes=2*m, planes=2*m, stride=1)
        self.layer7 = BasicBlock(in_planes=2*m, planes=2*m,  stride=1)
        self.layer8 = AdjustBlock(in_planes=2*m, planes=4*m, stride=2)
        self.layer9 = BasicBlock(in_planes=4*m, planes=4*m,  stride=1)
        self.layer10 = BasicBlock(in_planes=4*m, planes=4*m,  stride=1)
        self.bn = nn.BatchNorm2d(4*m)
        
        self.linear = nn.Linear(4*m*1*1, num_classes)
    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
cnn = CNN(num_classes=9, m=2)
opimizer = optim.Adam(cnn.parameters(), lr=lr, weight_decay=0.01)
loss_func = nn.CrossEntropyLoss()

viz = Visdom()
viz.line([[0.0, 0.0, 0.0]], [0.], win='loss', opts=dict(title='loss', legend=['train_loss', 'val_loss', 'val_acc']))
global_step = 0
#train and test
for epoch in range(epochs):
    for step,(x,y) in enumerate(train_loader):
        cnn.train()
        b_x = Variable(x)
        b_y = Variable(y)

        out = cnn(b_x)
        loss = loss_func(out, b_y.long())
        opimizer.zero_grad()
        loss.backward()
        opimizer.step()

        global_step += 1
        val_acc, val_loss = evalute(cnn, val_loader)
        viz.line([[loss.item(), val_loss, val_acc]], [global_step], win='loss', update='append')
        
        if step % 20 == 0:
            print('Epoch:',epoch,'|train loss:'+str(loss.item()))


