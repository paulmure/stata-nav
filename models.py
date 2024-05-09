import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Baseline(torch.nn.Module):

    def __init__(self, num_classes):
        super(Baseline, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_layers = nn.Sequential(*resnet_layers)
        for param in self.resnet_layers.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


class ReplaceLastLayer(torch.nn.Module):

    def __init__(self, num_classes):
        super(ReplaceLastLayer, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]

        for layer in resnet_layers[:7]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in resnet_layers[7:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.resnet_layers = nn.Sequential(*resnet_layers)

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


class ReplaceAll(torch.nn.Module):

    def __init__(self, num_classes):
        super(ReplaceAll, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_layers = nn.Sequential(*resnet_layers)
        for param in self.resnet_layers.parameters():
            param.requires_grad = True

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


class KaimingUniform(torch.nn.Module):

    def __init__(self, num_classes):
        super(KaimingUniform, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        resnet = resnet50()
        resnet.apply(init_weights)
        resnet_layers = list(resnet.children())[:-1]
        self.resnet_layers = nn.Sequential(*resnet_layers)
        for param in self.resnet_layers.parameters():
            param.requires_grad = True

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x
