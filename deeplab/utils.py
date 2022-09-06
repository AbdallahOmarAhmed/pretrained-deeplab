import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Pooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.relu(self.bn(self.conv(self.pool(x))))
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, dilation=6, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, dilation=12, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, dilation=18, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)

        self.pool = Pooling(in_ch, out_ch)

        self.conv_end = nn.Conv2d(out_ch*5, out_ch, 1, bias=False)
        self.bn_end = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.pool(x)
        return self.relu(self.bn_end(self.conv_end(torch.cat([x1, x2, x3, x4, x5], dim=1))))


def dice_loss(input, target):
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total