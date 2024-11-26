import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss

# class CrossEntropyLoss1(nn.Module):
#     def __init__(self):
#         super(CrossEntropyLoss1, self).__init__()
#         self.ce_criterion = nn.BCELoss()

#     def forward(self, logits, label):
#         label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
#         loss = -torch.mean(torch.sum(label * F.log(logits, dim=1), dim=1), dim=0)
#         return loss

class CategoryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        # import pdb; pdb.set_trace()
        label = label / torch.sum(label, dim=-1, keepdim=True)
        loss = -1.0 * torch.sum(label * torch.log_softmax(pred, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss

class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):   # [B,T]
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)

        return first_term + second_term

class GeneralizedCE_Mask(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE_Mask, self).__init__()

    def forward(self, logits, label, mask):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label * mask, dim=1) + 1e-7
        neg_factor = torch.sum((1 - label) * mask, dim=1) + 1e-7

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label * mask, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label) * mask, dim=1)/neg_factor)

        return first_term + second_term

class BCE(nn.Module):
    def __init__(self, ):
        super(BCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        first_term = - torch.mean(torch.sum((logits + 1e-7).log() * label, dim=1) / pos_factor)
        second_term = - torch.mean(torch.sum((1 - logits + 1e-7).log() * (1 - label), dim=1) / neg_factor)

        return first_term + second_term


class BCE1(nn.Module):
    def __init__(self, ):
        super(BCE1, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        factor = torch.sum(label, dim=1) + 1e-7
        loss = - torch.mean(torch.sum((logits + 1e-7).log() * label, dim=1) / factor)

        return loss

class Focal(nn.Module):
    def __init__(self, gamma):
        super(Focal, self).__init__()
        self.gamma = gamma

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        factor = torch.sum(label, dim=1) + 1e-7
        loss = - torch.mean(torch.sum((logits + 1e-7).log() * label * ((1 - logits)**self.gamma), dim=1) / factor)

        return loss
