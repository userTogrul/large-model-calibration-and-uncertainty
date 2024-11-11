import torch
import torch.nn as nn
import torch.nn.functional as F

# bald score for uncertainty
def BALDn(mu, var, pred, gt):
    # get predicted labels by converting to pred prob
    pred_sm = torch.argmax(F.softmax(pred, dim=-1), dim=-1)
    pred_lab = F.one_hot(pred_sm, num_classes=pred.shape[-1]).float()
    gt_lab = F.one_hot(gt.to(torch.int64), num_classes=pred.shape[-1]).float()

    true_mask = torch.sum(pred_lab * gt_lab, dim=-1)
    mu_se = torch.sum(pred_lab * mu, dim=-1)
    var_se = torch.sum(pred_lab * var, dim=-1)

    val1 = mu_se / (torch.sqrt(var_se ** 2 + 1) + 1e-8)
    dist = torch.distributions.Normal(mu_se, var_se)
    y1 = dist.cdf(val1)
    
    hy1 = -y1 * torch.log(y1 + 1e-6) - (1.0 - y1) * torch.log(1.0 - y1 + 1e-6)
    baldmap = hy1 - torch.exp(-mu_se ** 2/ 2.0 / (var_se ** 2 + 1.0)) / torch.sqrt(var_se ** 2 + 1.0)
    baldmap = torch.sigmoid(baldmap)

    score = baldmap * 0.5 * true_mask + (2.0 -baldmap) * 0.5 * (1-true_mask) + 1.0

    return score
