import torch
import torch.nn as nn
import numpy as np


class hard_concrete():
    def __init__(self):
        self.eps = 1e-20
        self.sig = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(0, 1)
        self.gamma = -0.1
        self.zeta = 1.1
        self.beta = 0.66
        self.const1 = self.beta * np.log(-self.gamma / self.zeta + self.eps)

    def l0_train(self, logAlpha):
        U = torch.rand(logAlpha.size()).cuda()
        s = self.sig((torch.log(U + self.eps) - torch.log(1 - U + self.eps) + logAlpha + self.eps) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = self.hardtanh(s_bar)
        return mask

    def l0_test(self, logAlpha):
        s = self.sig(logAlpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = self.hardtanh(s_bar)
        return mask

    def get_loss2(self, logAlpha):
        return self.sig(logAlpha - self.const1)
