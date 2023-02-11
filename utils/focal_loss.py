# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Below code are based on
# https://zhuanlan.zhihu.com/p/28527749

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):

        alpha = torch.zeros_like(targets)
        self.alpha = self.alpha.to(inputs.device)
        for ii in range(self.class_num):
            alpha[targets==ii] = self.alpha[ii]

        probs = (inputs) * targets + (1 - inputs) * (1 - targets)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*torch.log(probs)

        return batch_loss