import torch
import torch.nn as nn

class simam_module2(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module2, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam2"

    def forward(self, x):  #x.shape B D n --D为通道数 k为点数
        b,d,k = x.size()
        n = k - 1
        x_minus_mu_square = (x - x.mean(dim=[2], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
