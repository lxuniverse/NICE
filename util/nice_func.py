import torch.nn as nn
from .hc_func import hard_concrete
from util.models import Net


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.hc = hard_concrete()

    def forward(self, x):
        logAlpha = self.conv1(x)
        mask = self.hc.l0_train(logAlpha)
        loss2 = self.hc.get_loss2(logAlpha)
        return mask, loss2

    def evaluate(self, x):
        logAlpha = self.conv1(x)
        mask = self.hc.l0_test(logAlpha)
        return mask


class NICE(nn.Module):
    def __init__(self):
        super(NICE, self).__init__()
        self.cnn = Net()
        self.saliency_l = Generator()

    def forward(self, x):
        z, loss2 = self.saliency_l(x)
        rationale = x * z
        out = self.cnn(rationale)
        return out, z, loss2
