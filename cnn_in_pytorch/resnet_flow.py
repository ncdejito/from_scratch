
# Notes on program flow in PyTorch Resnet Implementation
# comments are layers in nn.Sequential

model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):

x = self.conv1(x)
# nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
x = self.bn1(x)
# nn.BatchNorm2d(64)
x = self.relu(x)
# nn.ReLU(inplace=True)
x = self.maxpool(x)
# nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

x = self.layer1(x)
self.layer1 = self._make_layer(block, 64, layers[0])
def _make_layer(self, block, planes, blocks, stride=1)
self.inplanes != planes * block.expansion
self.inplanes = 64 # resnet defined
planes = 64
class BasicBlock expansion = 1
def __init__(self, inplanes, planes, stride=1, downsample=None)
(64 != 64*1) == False
layers.append(block(self.inplanes, planes, stride, downsample))
downsample = None
# BasicBlock(64, 64, stride=1, downsample = None)
self.inplanes = planes * block.expansion
for _ in range(1, blocks):
layers.append(block(self.inplanes, planes))
# BasicBlock(64, 64, stride=1, downsample = None)

x = self.layer2(x)
self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
def _make_layer(self, block, planes, blocks, stride=2)
stride != 1
# downsampleL2 = nn.Sequential(
#     nn.Conv2d(64, 128*1, kernel_size=1, stride=2, bias=False)
#     ,nn.BatchNorm2d(128*1)
#             )
layers.append(block(self.inplanes, planes, stride, downsample))
# BasicBlock(64, 128, stride=2, downsample = downsampleL2)
self.inplanes = planes * block.expansion
128 = 128*1
layers.append(block(self.inplanes, planes))
# BasicBlock(128, 128, stride=1, downsample = None)

x = self.layer3(x)
self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
def _make_layer(self, block, planes, blocks, stride=2)
stride != 1
# downsampleL3 = nn.Sequential(
#     nn.Conv2d(128, 256*1, kernel_size=1, stride=2, bias=False)
#     ,nn.BatchNorm2d(256*1)
#             )
layers.append(block(self.inplanes, planes, stride, downsample))
# BasicBlock(128, 256, stride=2, downsample = downsampleL3)
self.inplanes = planes * block.expansion
256 = 256*1
layers.append(block(self.inplanes, planes))
# BasicBlock(256, 256, stride=1, downsample = None)

x = self.layer4(x)
self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
def _make_layer(self, block, planes, blocks, stride=2)
stride != 1
# downsampleL4 = nn.Sequential(
#     nn.Conv2d(256, 512*1, kernel_size=1, stride=2, bias=False)
#     ,nn.BatchNorm2d(512*1)
#             )
layers.append(block(self.inplanes, planes, stride, downsample))
# BasicBlock(256, 512, stride=2, downsample = downsampleL4)
self.inplanes = planes * block.expansion
512 = 512*1
layers.append(block(self.inplanes, planes))
# BasicBlock(512, 512, stride=1, downsample = None)

# Below layers replaced by fastai
x = self.avgpool(x)
# nn.AdaptiveAvgPool2d((1, 1))
x = x.view(x.size(0), -1)
# Flatten()
x = self.fc(x)
