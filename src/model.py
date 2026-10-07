import torch
import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self,n_classes):
        super(Mymodel, self).__init__()#继承nn.Module必写
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)#kernel_size=5, padding=2让特征图大小不变
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)#kernel_size=3, padding=1让特征图大小不变
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)#下一层 = （上一层+1）/2
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)#Dense = 全连接层
        self.fc8 = nn.Linear(4096, n_classes)#只有5种花，所以不用图中的18
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)#压成一维进全连接层
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)#只有这个不加激活函数，不然会失去表达负数分的能力
        return x
