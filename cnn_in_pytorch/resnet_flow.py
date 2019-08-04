
# model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):

# x = self.conv1(x)
nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
# x = self.bn1(x)
nn.BatchNorm2d(64)
# x = self.relu(x)
nn.ReLU(inplace=True)
# x = self.maxpool(x)
nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

# x = self.layer1(x)
self.layer1 = self._make_layer(block, 64, layers[0])
def _make_layer(self, block, planes, blocks, stride=1)
self.inplanes != planes * block.expansion
self.inplanes = 64
planes = 64
class BasicBlock expansion = 1
(64 != 64*1) == False
layers.append(block(self.inplanes, planes, stride, downsample))
downsample = None
class BasicBlock(nn.Module):
def __init__(self, inplanes, planes, stride=1, downsample=None):
(64, 64, stride=1, downsample = None)
identity = x
# out = self.conv1(x)
self.conv1 = conv3x3(inplanes, planes, stride)
def conv3x3(in_planes, out_planes, stride=1):
nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1, bias=False)
# out = self.bn1(out)
nn.BatchNorm2d(64)
# out = self.relu(out)
nn.ReLU(inplace=True)
# out = self.conv2(out)
conv3x3(planes, planes)
nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1, bias=False)
# out = self.bn2(out)
nn.BatchNorm2d(64)
# out += identity

# out = self.relu(out)
nn.ReLU(inplace=True)

x = self.layer2(x)
x = self.layer3(x)
x = self.layer4(x)

x = self.avgpool(x)
x = x.view(x.size(0), -1)
x = self.fc(x)