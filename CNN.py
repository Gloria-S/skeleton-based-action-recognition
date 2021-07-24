import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

class my_dataset(Dataset):
    def __init__(self,split,ifAugumentation=False):
        self.split=split
        self.skeleton_list=[]
        self.label_list=[]
        if self.split=='train':
            if ifAugumentation:  # 数据增强
                self.filename="train_label path_add.txt"
            else:
                self.filename="train_label path.txt"
        else:
            self.filename="test_label path.txt"
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split('\n')[0].split(' ')
                # print(temp)
                self.label_list.append(int(temp[0]))
                sample = np.load(temp[1]).transpose((0, 4, 2, 3, 1))  # batchsize，单双人，帧数，关节，坐标
                # print(sample.shape)
                # sample = np.pad(sample, ((0, 0), (0, 0), (0, 320 - sample.shape[2]), (0, 0), (0, 0)))
                sample=sample[:,:,0:40,:,:]
                self.skeleton_list.append(sample[0][0])
                if sample[0][1].all() != 0.0:
                    self.skeleton_list.append(sample[0][1])


    def __getitem__(self, item):
        sample=self.skeleton_list[item]
        label = self.label_list[item]
        return sample, label

    def __len__(self):
        return len(self.skeleton_list)

####################模型定义##########
def define_model(D_in, H, D_out):
    # 使用torch.nn.Sequential定义序列化模型
    net = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    class Net(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
        def __init__(self):
            super(Net, self).__init__()
            # self.conv=nn.Conv2d(320, 40, 1)  # 降维
            self.conv1 = nn.Conv2d(40, 80, 2)
            #self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
            self.conv2 = nn.Conv2d(80, 60, 2)  # 同样是卷积层
            self.fc1 = nn.Linear(36 * 5 * 5, 200)  # 接着两个全连接层 修改后的正确的代码
            self.fc2 = nn.Linear(200, 80)

        def forward(self, x):
            # print(x.size())
            x = F.relu(self.conv1(x))
            # print(x.size())
            x = F.relu(self.conv2(x))
            # print(x.size())
            x = x.view(-1, 36 * 5 * 5)  # 修改后的正确的代码
            # print(x.size())

            x = F.relu(self.fc1(x))
            # print(x.size())
            #x = F.relu(self.fc2(x))
            x = self.fc2(x)
            # print(x.size())
            return x

    net = Net()
    return net

################损失函数定义#########
def define_loss():
    Loss = nn.CrossEntropyLoss()
    return Loss


##############优化器定义#############
def define_optimizer():
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    return optimizer


###################模型训练#########
def train(train_loader, net, Loss, optimizer,epoch):
    running_loss, running_acc = 0.0, 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img=img.to(torch.float32)
        label=label.long()
        out = net(img)
        loss = Loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        running_acc += (predicted == label).sum().item()
        print('Epoch [{}/5], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, i, len(train_loader), loss.item(), (predicted==label).sum().item()/40))

    return net, running_loss, running_acc


###################模型测试#########
def test(test_loader, net, Loss, running_loss, running_acc):
    #net = torch.load(net_path)
    test_loss, test_acc = 0.0, 0.0
    for i, data in enumerate(test_loader):
        img, label = data
        label = list(label)
        label = torch.tensor(label)
        img = img.to(torch.float32)
        label = label.long()

        out = net(img)
        loss = Loss(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss += loss.item() * label.size(0)

        _, predicted = torch.max(out, 1)

        test_acc += (predicted == label).sum().item()

    print("Train {} epoch, Loss: {:.6f}, Acc: {:.6f}, Test_Loss: {:.6f}, Test_Acc: {:.6f}".format(
    epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset)),
    test_loss / (len(test_dataset)), test_acc / (len(test_dataset))))
    return 0

if __name__ == '__main__':
    split = 'train'
    train_dataset = my_dataset(split, True)
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=1)


    N, D_in, H, D_out = 4, 17*3*40, 100, 4 # 产生随机输入和输出张量

    net = define_model(D_in, H, D_out)
    Loss = define_loss()
    optimizer = define_optimizer()

    split = 'test'
    test_dataset = my_dataset(split,True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=1)


    for epoch in range(0,5):
        net, running_loss, running_acc = train(train_loader, net, Loss, optimizer, epoch)
        #net_path = 'D:/作业pycharm/姿态分类/net.pth'
        #torch.save(net, net_path)
        test(test_loader, net, Loss, running_loss, running_acc)
