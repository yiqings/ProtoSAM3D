import torch
import torch.nn as nn
import torch.nn.functional as F

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))


class ProjectionHead_3D(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead_3D, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv3d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))

def BNReLU(num_features, bn_type=None, **kwargs):
    if bn_type == 'torchbn':
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'torchsyncbn':
        return nn.Sequential(
            nn.SyncBatchNorm(num_features, **kwargs),
            nn.ReLU()
        )