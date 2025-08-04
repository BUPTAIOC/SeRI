import torch.nn as nn


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()

        # 第一层卷积：1x28x28 -> 16x26x26
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # 池化：16x26x26 -> 16x13x13
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积：16x13x13 -> 32x11x11
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        # 再池化：32x11x11 -> 32x5x5
        # 输出特征的总量：32 * 5 * 5 = 800

        # 全连接层直接输出
        self.fc = nn.Linear(800, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平成一维
        x = x.view(x.size(0), -1)

        # 全连接分类
        x = self.fc(x)
        return x


class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()

        # 仅一层卷积：1x28x28 -> 8x26x26
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

        # 最大池化：8x26x26 -> 8x13x13
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积：8x13x13 -> 16x11x11
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        # 最大池化：16x11x11 -> 16x5x5
        # 全局平均池化：16x5x5 -> 16
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 最后一层线性分类：16 -> 10
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 全局平均池化后展平
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # 分类层
        x = self.fc(x)
        return x
