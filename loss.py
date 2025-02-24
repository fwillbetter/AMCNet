import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class ClassificationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):

        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            return NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        """
        计算交叉熵损失函数

        Args:
            logit (torch.Tensor): 经过神经网络计算得到的预测结果，shape为(n, c, h, w)
            target (torch.Tensor): 真实标签，shape为(n, h, w)

        Returns:
            torch.Tensor: 损失值
        """
        n, c, h, w = logit.shape
        criterion = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index,
            size_average=self.size_average
        )
        if self.cuda:
            criterion = criterion.to('cuda')

        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        """
        计算Focal Loss函数

        Args:
            logit (torch.Tensor): 经过神经网络计算得到的预测结果，shape为(n, c, h, w)
            target (torch.Tensor): 真实标签，shape为(n, h, w)
            gamma (float): 控制缩放因子的参数，默认为2
            alpha (float): 控制缩放因子的参数，默认为0.5

        Returns:
            torch.Tensor: 损失值
        """
        n, c, h, w = logit.shape
        criterion = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index,
            size_average=self.size_average
        )

        if self.cuda:
            criterion = criterion.to('cuda')

        logpt = - criterion(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
