import torch
from torch import nn
from einops import einsum, rearrange
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        mean = 0
        std = (2/(in_features+out_features)) ** 0.5
        lower, upper = -3 * std, 3 * std
        W = torch.rand(out_features, in_features, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, mean, std, lower, upper)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        W = torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, 0, 1, -3, 3)
        self.W = nn.Parameter(W)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    
class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = (((1/self.d_model)*(x**2).sum(dim=-1, keepdim=True))+self.eps)**0.5
        res = x/rms
        
        res = res * self.gain
        return res.to(in_dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = math.floor(8/3 * d_model / 64) * 64
        self.W1 = nn.Parameter(torch.rand(d_ff, d_model))
        self.W2 = nn.Parameter(torch.rand(d_model, d_ff))
        self.W3 = nn.Parameter(torch.rand(d_ff, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SiLU = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        SiLU = SiLU * torch.sigmoid(SiLU)
        xW3 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        return einsum((SiLU * (xW3)), self.W2 , "... d_ff, d_model d_ff-> ... d_model")
    
def softmax(t: torch.Tensor, i):
    sub = torch.max(t, dim=i, keepdim=True).values
    res = t - sub
    res = res.exp()/res.exp().sum(dim=i, keepdim=True)
    return res
    
def scaled_dot_product_attention(Q: torch.Tensor, K, V, mask):
    S = softmax((einsum(Q, K, "... n d_k, ... m d_k ->... n m")/(Q.shape[-1]**0.5)).masked_fill(~mask, -torch.inf), -1)
    return S @ V

def MultiHead(Q, K, V, mask, num_heads):
    return scaled_dot_product_attention(Q, K, V, mask)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, d_model, d_k, d_v):
        super().__init__()
        self.Wq = nn.Parameter(torch.rand(heads, d_k, d_model))
        self.Wk = nn.Parameter(torch.rand(heads, d_k, d_model))
        self.Wv = nn.Parameter(torch.rand(heads, d_v, d_model))
        self.Wo = nn.Parameter(torch.rand(d_model, heads*d_v))
        self.num_heads = heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.triu(torch.ones(x.shape[-2], x.shape[-2], device=x.device)).T == 1
        Q = einsum(self.Wq, x, "heads d_k d_model, ... num_sequence d_model -> ... heads num_sequence d_k")
        K = einsum(self.Wk, x, "heads d_k d_model, ... num_sequence d_model -> ... heads num_sequence d_k")
        V = einsum(self.Wv, x, "heads d_v d_model, ... num_sequence d_model -> ... heads num_sequence d_v")
        hd_v = rearrange(MultiHead(Q, K, V, mask, self.num_heads), "... heads num_sequence d_v -> ... num_sequence (heads d_v)")
        return einsum(self.Wo, hd_v, "d_model hd_v, ... num_sequence hd_v -> ... num_sequence d_model")
        


    
    