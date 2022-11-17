import numpy as np
from torch import nn

from src.VariationalBottleneck import VariationalBottleneck

class SMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class VBSMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.VB = VariationalBottleneck((width,))
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x, eps=None):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.VB(x, eps)
        x = self.l3(x)
        return x

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()


class DMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.l5 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        return x


class VBDMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.VB = VariationalBottleneck((width,))
        self.l5 = nn.Linear(width, num_classes)

    def forward(self, x, eps=None):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.VB(x, eps)
        x = self.l5(x)
        return x

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()


class LeNetR(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super().__init__()
        act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(inplace=True),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(inplace=True),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class VBLeNetR(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super().__init__()
        act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(inplace=True),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(inplace=True),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(inplace=True),
        )
        self.VB = VariationalBottleneck((hideen,))
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )
        
    def forward(self, x, eps=None):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.VB(out, eps)
        out = self.fc(out)
        return out

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()
    

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class VBLeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.VB = VariationalBottleneck((hideen,))
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )
        
    def forward(self, x, eps=None):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.VB(out, eps)
        out = self.fc(out)
        return out

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()