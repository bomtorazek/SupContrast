"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class CrossSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CrossSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features_A, features_B, labels_A=None, labels_B=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j from 'B'
                has the same class as sample i from 'A'. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features_A.is_cuda and features_B.is_cuda
                  else torch.device('cpu'))

        if len(features_A.shape) < 3 or len(features_B.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        if len(features_A.shape) > 3:
            features_A = features_A.view(features_A.shape[0], features_A.shape[1], -1)
        if len(features_B.shape) > 3:
            features_B = features_B.view(features_B.shape[0], features_B.shape[1], -1)


        batch_size = features_A.shape[0]
        if labels_A is not None and labels_B is not None:
            exist_label = 1
        elif labels_A is None and labels_B is None:
            exist_label = 0
        else:
            raise ValueError('both labels_A and labels_B should be None or not')


        if exist_label and mask is not None: # 1 1
            raise ValueError('Cannot define both `labels` and `mask`')
        elif exist_label == 0 and mask is None: # 0 0unsupervised loss
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif exist_label: # 1 0
            labels_A = labels_A.contiguous().view(-1, 1)
            labels_B = labels_B.contiguous().view(-1, 1)
            if labels_A.shape[0] != batch_size or labels_B.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels_A, labels_B.T).float().to(device)
            mask_check = torch.eq(labels_B, labels_A.T).float().to(device)
            assert mask == mask_check.T #FIXME
        else: # 0 1
            mask = mask.float().to(device)

        contrast_count = features_A.shape[1] # n_view (augment) of feature
        contrast_feature_A = torch.cat(torch.unbind(features_A, dim=1), dim=0)
        contrast_feature_B = torch.cat(torch.unbind(features_B, dim=1), dim=0) 
        # make (bsz*n_view, -1), bsz is inner

        if self.contrast_mode == 'one':
            anchor_feature = features_A[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature_A
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature_B.T),
            self.temperature) # zi * zp mat with size of (bsz*n_view, bsz*n_view) 
                              # asymmetric for A and B
                              #  (A*B.T) = (B*A.T)

        # for numerical stability, want to avoid very large logits that lead to NaN problem
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() 
        # 어차피 빼도 gradient는 똑같음.

        # tile mask, cause originally shape is (bsz * bsz)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )  # DO NOT NEED THIS ONE IN CROSS DOMAIN

        # input, dim, index, src
        # document 보면 이해는 되는데 너무 어렵다.
        # src랑 index는 겹쳐서 같은 위치를 보면 된다. (예를 들어 (1,2)) src가 스칼라면 그걸 index랑 같은 shape으로 보자.
        # 같은 위치의 index의 값을 참조하여 위치에서의 dim만 그 값으로 바꾼다. ((1,2)의 index가 3이고 dim이 1이면 1,3 으로 바뀜)
        # input의 (1,3)을 src의 (1,2로 대체)
        # 이거 그냥 ones에서 diagonal만 0으로 바꾸는 건데?

        # mask = mask * logits_mask # label_mask (diagonal 1) * logits_mask ( diagonal 0) 

        # compute log_prob
        exp_logits = torch.exp(logits)  # * logits_mask # self-cont filtering
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # logits: numerator of (2), positive will be considered further by mask
        # torch.log: denominator of (2)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean() # n_view, batch_size

        return loss
