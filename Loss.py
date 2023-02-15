#segmentation loss:#

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import numpy as np


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

#new losses

class DiceLoss(nn.Module):
    def __init__(self, ignore=255):
        super().__init__()
        self.smooth = 1
        self.ignore = ignore

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        #print(preds.size())
        preds = preds[targets != self.ignore]
        targets = targets[targets != self.ignore]
        iflat = preds.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()

        #p = nn.functional.softmax(preds, dim=-1)

        return 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() + self.smooth))  #- 1 * torch.sum(p * nn.functional.log_softmax(preds, dim=-1))

class Adversarial_loss(nn.Module):#搞一个对比，跟CrossEntropy

    def __init__(self):
        super(Adversarial_loss, self).__init__()

    def forward(self, pred, gpu):
        N, C, H, W = pred.size()
        # sg = nn.Sigmoid()
        bce = nn.BCEWithLogitsLoss()
        z = torch.ones(N, C, H, W)
        #z = Variable(z).cuda(gpu)
        z = z.to(gpu)
        out = bce(pred, z)

        # D = sg(pred)
        # D = torch.clamp(D, min = 1e-9, max = 1-(1e-9))
        # temp = D.view(N, 1, -1)#N,1,H*W
        # #print(temp.size())
        # temp = temp.log()
        # out = - temp.sum() / (H * W)
        return out

class Reconstruction_loss(nn.Module):

    def __init__(self):
        super(Reconstruction_loss, self).__init__()

    def forward(self, pred, target):
        #_, C, H, W = target.size()
        mse = nn.MSELoss()
        #sm = torch.softmax()

        #pred = sm(pred)
        #temp = torch.pow((target - pred), 2)
        #out = temp.sum() / (C * H * W)
        out = mse(pred, target)
        return out
    

class DisCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(DisCrossEntropyLoss, self).__init__()

    def forward(self, pred, z, gpu):
        N, C, H ,W = pred.size()
        bce = nn.BCEWithLogitsLoss()

        if z == 1:
            z = torch.ones(N, C, H, W)
        elif z == 0:
            z = torch.zeros(N, C, H, W)
        #z = Variable(z).cuda(gpu)
        z = z.to(gpu)
        temp = bce(pred, z)
        out = temp#too small
        return out

class CRloss(nn.Module):
    def __init__(self):
        super(CRloss, self).__init__()

    def forward(self, a, b, c, gpu):
        ranking = nn.MarginRankingLoss(1)
        temp1 = nn.functional.pairwise_distance(a, c)
        temp2 = nn.functional.pairwise_distance(b ,c)
        c = torch.ones(temp1.size())
        c = c.to(gpu)
        # out = max(c, temp)
        out = ranking(temp1, temp2, c)
        return out

class CRFloss(nn.Module):
    def __init__(self):
        super(CRFloss, self).__init__()

    def forward(self, a, b, gpu):
        ranking = nn.MarginRankingLoss(1)
        c = torch.ones(a.size())
        c = c.to(gpu)
        # out = max(c, temp)
        out = ranking(a, b, c)
        return out

# p_logit: [batch,class_num]
class em_loss(nn.Module):
    def __init__(self):
        super(em_loss, self).__init__()

    def forward(self, p_logit):
        p = nn.functional.sigmoid(p_logit)
        return -1 * torch.sum(p * nn.functional.sigmoid(p_logit)) / (p_logit.size()[2] * p_logit.size()[3])

class cosis_loss(nn.Module):#buyong
    def __init__(self):
        super(cosis_loss, self).__init__()

    def forward(self, a, b, gpu):
        weights = torch.tensor([1, 1], dtype=torch.float).to(gpu)
        out = -1 * weights * (b*(a.log()) + torch.sum(-1* b, 1)*torch.sum(-1*a, 1).log())
        return out

class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, p_logit):
        weights = torch.tensor([1, 1], dtype=torch.float)
        sg = nn.Sigmoid()
        p = sg(p_logit)
        out = -1 * p * p.log()
        out = out.sum() / (p_logit.size()[2] * p_logit.size()[3])
        return out

class CBST_loss(nn.Module):
    def __init__(self):
        super(CBST_loss, self).__init__()

    def forward(self, pred, y_hat, k):
        out = - y_hat * pred.log() - k * y_hat# torch.norm(y_hat, p=1)#
        # print(out)
        out = out.sum()/(pred.size(2)*pred.size(3))
        return out

class ACLoss3DV2(nn.Module):

    def __init__(self, miu=1.0, classes=3):
        super(ACLoss3DV2, self).__init__()

        self.miu = miu
        self.classes = classes

    def forward(self, predication, label):
        min_pool_x = nn.functional.max_pool3d(
            predication * -1, (3, 3, 3), 1, 1) * -1
        contour = torch.relu(nn.functional.max_pool3d(
            min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)

        # length
        length = torch.sum(torch.abs(contour))

        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        return region + length