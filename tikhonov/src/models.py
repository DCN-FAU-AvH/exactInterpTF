import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, d, h, tau, causal=False):
        super().__init__()
        self.d = d
        self.h = h
        self.tau = tau
        self.to_q = nn.Linear(d, h, bias=False)
        self.to_k = nn.Linear(d, h, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.causal = causal
    def forward(self, x):
        _, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
            
        scores = q @ k.transpose(-2, -1)

        ## Softmax self-attention:
        if self.causal:
            ## Causal self-attention:
            mask = torch.triu(torch.ones(n, n, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        scores = scores / self.tau
        attn = torch.softmax(scores, dim=-1)
        
        out = attn @ v
        return out

class TFBlock(nn.Module):
    def __init__(self, d, h, tau, causal = False):
        super().__init__()

        self.rho = nn.Parameter(torch.zeros(1))  # Random initialization at 0
        self.eta = nn.Parameter(torch.zeros(1))   # Initialize at 0

        self.attn = SelfAttentionLayer(d, h, tau, causal)

        self.ff = nn.Sequential(
            nn.Linear(d, 4),
            nn.ReLU(),
            nn.Linear(4, d, bias = False)
        )

    def forward(self, x):
        x = self.rho * x + self.attn(x)
        x = self.eta * x + self.ff(x)
        return x
    
class TF(nn.Module):
    def __init__(self, d, h, tau, causal, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[TFBlock(d, h, tau, causal) for _ in range(depth)])
    def forward(self, x):
        x = self.blocks(x)
        return x 