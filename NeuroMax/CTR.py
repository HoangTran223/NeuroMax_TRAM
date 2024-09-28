import torch
from torch import nn


class CTR(nn.Module):
    def __init__(self, weight_loss_CTR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_CTR = weight_loss_CTR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, a, b, M):
        # M: KxV
        # a: BxK
        # b: BxK

        # print(a)
        # print(b)
        # print(M)

        # print(M)
        # print(M.max())

        if self.weight_loss_CTR <= 1e-6:
            return 0.0
        
        B, K = a.size()
        M = M.unsqueeze(0).repeat(B, 1, 1)

        device = M.device


        # Sinkhorn's algorithm
        u = (torch.ones_like(a) / K).to(device)  # KxB

        K_mat = torch.exp(-M * self.sinkhorn_alpha)  # KxVxB

        err = float('inf')
        cpt = 0

        while err > self.stopThr and cpt < self.OT_max_iter:
            # Update v: v = b / (K^T u)

            KTu = torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(2)).squeeze(2)  # Shape: (B, V)
            v = b / (KTu + self.epsilon)  # Shape: (B, V)
            # Update u: u = a / (K v)

            Kv = torch.bmm(K_mat, v.unsqueeze(2)).squeeze(2)  # Shape: (B, K)
            u = a / (Kv + self.epsilon)  # Shape: (B, K)

            cpt += 1
            if cpt % 50 == 1:
                # Compute the marginal constraint error
                KTu = torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(2)).squeeze(2)  # Shape: (B, V)
                bb = v * KTu  # Shape: (B, V)
                err = torch.max(torch.sum(torch.abs(bb - b), dim=1)).item()


        # Transport matrix for the batch
        transp = u.unsqueeze(2) * K_mat * v.unsqueeze(1)  # Shape: (B, K, V)

        loss_CTR = torch.mean(torch.sum(transp * M, dim=(1, 2)))  # Shape: (B,) -> mean (1,)
        loss_CTR *= self.weight_loss_CTR

        return loss_CTR
