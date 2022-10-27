import torch
import torch.nn.functional as F
from torch import nn

class DropBlockT_1d(nn.Module):
    def __init__(self, in_features=64, block_size=41):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 1
        self.block_size = block_size
        self.in_features = in_features
        self.fc_1 = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
            )
        self.fc_2 = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        input_attention_mean = torch.mean(torch.mean(input, dim=3), dim=1).detach()  # 32 64
        input_attention_max = torch.max(input, dim=3)[0]
        input_attention_max = torch.max(input_attention_max, dim=1)[0]
        avg_out = self.fc_1(input_attention_mean)
        max_out = self.fc_2(input_attention_max)
        out = avg_out + max_out
        input_attention_out = self.sigmoid(out).view(n, 1, t)
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        input_a = input1 * input_attention_out
        input_a = input_a.view(n, c, v, t).permute(0, 1, 3, 2)

        input_abs = torch.mean(torch.mean(torch.abs(input_a), dim=3), dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n, 1, t)
        gamma = 0.0028
        input_a = input_a.permute(0, 1, 3, 2).contiguous().view(n, c*v, t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1, c*v, 1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        return (input_a * mask * mask.numel() /mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)
