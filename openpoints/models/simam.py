import torch
import torch.nn as nn


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):  #fj.shape B D S K  S为采样点数  K为邻近点数
        b, d, s, k = x.size()
       # b, c, h, w = x.size()
        n = k - 1
        #n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



