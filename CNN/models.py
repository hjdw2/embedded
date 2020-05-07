'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet_local(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet_local, self).__init__()
        self.LC_layer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  #stride=1, 2p-kernel + 1 =0
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.LC_layer_out = nn.Linear(8*8*128, num_classes)

    def forward(self, x):
        x = self.LC_layer(x)
        x = x.view(-1, 8*8*128)
        x = self.LC_layer_out(x)
        return x

class ConvNet_lower(nn.Module):
    def __init__(self, in_channel):
        super(ConvNet_lower, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),  #stride=1, 2p-kernel + 1 =0
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  #stride=1, 2p-kernel + 1 =0
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        return out

class ConvNet_upper(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet_upper, self).__init__()
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  #stride=1, 2p-kernel + 1 =0
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  #stride=1, 2p-kernel + 1 =0
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer_out = nn.Linear(8*8*128, num_classes)

    def forward(self, x):
        out = self.conv_layer3(x)
        out = self.conv_layer4(out)
        out = out.view(-1, 8*8*128)
        out = self.layer_out(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_lower(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_lower, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class ResNet_upper(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_upper, self).__init__()
        self.in_planes = 128

        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return  out


def ResNet18_lower():
    return ResNet_lower(BasicBlock, [2,2])

def ResNet18_upper(num_classes):
    return ResNet_upper(BasicBlock, [2,2], num_classes=num_classes)

def ResNet34_lower():
    return ResNet_lower(BasicBlock, [3,4,6,3])

def ResNet34_upper(num_classes):
    return ResNet_upper(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50_lower():
    return ResNet_lower(Bottleneck, [3,4,6,3])

def ResNet50_upper(num_classes):
    return ResNet_upper(Bottleneck, [3,4,6,3], num_classes=num_classes)

def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
