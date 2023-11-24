import torch
from torch import nn
from torchvision.models import squeezenet1_0, vgg11


class SqueezeNet:
    def __init__(self, weights=None):
        # Load weights and modify the classifier to have 4 output classes
        self.net = squeezenet1_0(weights='DEFAULT')
        self.net.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1))

        if weights:
            self.net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    def get_model(self):
        return self.net


class VGG11:
    def __init__(self, weights=None):
        # Load weights and modify the classifier to have 4 output classes
        self.net = vgg11(weights='DEFAULT')
        self.net.classifier[6] = nn.Linear(4096, 4)

        if weights:
            self.net.load_state_dict(torch.load(weights))

    def get_model(self):
        return self.net
