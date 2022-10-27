import torch
import torch.nn.functional as F
from torch import nn
import warnings


class DropBlock_Ske(nn.Module):
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point
        self.fc_1 = nn.Sequential(
                    nn.Linear(in_features=25, out_features=25, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=25, out_features=25, bias=True),
                )
        self.fc_2 = nn.Sequential(
                    nn.Linear(in_features=25, out_features=25, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=25, out_features=25, bias=True),
                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, keep_prob, A):  # n,c,t,v
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()

        input_attention_mean = torch.mean(torch.mean(input, dim=2), dim=1).detach()  # 32 25
        input_attention_max = torch.max(input, dim=2)[0].detach()
        input_attention_max = torch.max(input_attention_max, dim=1)[0]  # 32 25

        avg_out = self.fc_1(input_attention_mean)
        max_out = self.fc_2(input_attention_max)
        out = avg_out + max_out
        input_attention_out = self.sigmoid(out).view(n, 1, 1, self.num_point)
        input_a = input * input_attention_out

        input_abs = torch.mean(torch.mean(
            torch.abs(input_a), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        gamma = 0.024
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, min=0, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()
