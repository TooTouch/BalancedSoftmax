"""Code reference
https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification

Balanced Meta-Softmax for Long-Tailed Visual Recognition
Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li
NeurIPS 2020
"""

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_per_cls: list):
        super(BalancedSoftmax, self).__init__()
        self.num_per_cls = torch.tensor(num_per_cls)

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(labels=label, logits=input, num_per_cls=self.num_per_cls, reduction=reduction)


def balanced_softmax_loss(labels, logits, num_per_cls, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      num_per_cls: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    npc = num_per_cls.type_as(logits)
    npc = npc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + npc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss



def create_criterion(name: str, num_per_cls: list, params: dict = {}):
    if name == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(**params)
    elif name == 'BalancedSoftmax':
        criterion = BalancedSoftmax(num_per_cls=num_per_cls)
        
    return criterion
    