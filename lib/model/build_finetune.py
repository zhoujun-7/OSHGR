import torch
import timm
from timm.models.resnet import resnet50


net = timm.create_model('resnet18', pretrained=False)
net.fc.weight.data = net.fc.weight.data[:68]
net.fc.bias.data = net.fc.bias.data[:68]

# net.fc.weight.data[23:] = torch.randn_like(net.fc.weight.data[23:])
# net.fc.bias.data[23:] = torch.randn_like(net.fc.bias.data[23:])