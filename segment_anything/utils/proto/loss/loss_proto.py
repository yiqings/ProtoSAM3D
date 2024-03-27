from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.utils.proto.loss.loss_helper import FSAuxRMILoss, FSCELoss

class PPC(nn.Module, ABC):
    def __init__(self, configer):
        super(PPC, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, configer):
        super(PPD, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    """
    ProtoSeg对应的Loss
    """
    def __init__(self, configer=None,**kwargs):
        super(PixelPrototypeCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        print('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = self.configer.get('protoseg', 'loss_ppc_weight')
        self.loss_ppd_weight = self.configer.get('protoseg', 'loss_ppd_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')
        
        # self.num_classes = self.configer.get('data','num_classes')
        
        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)

    def forward(self, preds, target):
        # d, h, w = target.size(2), target.size(3), target.size(4)
        # 从这里开始target要变成适用于mask2former代码中关于loss计算的形式

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(seg, target.long())
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
            

        # seg = preds
        # pred = F.interpolate(input=seg, size=(d, h, w), mode='trilinear', align_corners=True)
        loss = self.seg_criterion(preds, target[0].squeeze(1).long())
        return loss