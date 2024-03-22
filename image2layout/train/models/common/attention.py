# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from einops import rearrange
from torch import einsum, nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim_q, dimvq, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim_q)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim_q, inner_dim, bias=False)
        self.to_kv = nn.Linear(dimvq, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim_q), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        # q:    [bs, heads, num_seq, dim_head]
        # k, v: [bs, heads, top_k, dim_head]

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


if __name__ == "__main__":
    # attn = Attention(256, 256)
    # memory = torch.randn(1, 330, 256)  # img: [bs, h*w, dim]
    # for dim_ref in [2, 4, 8, 16, 32]:
    #     FF = FeedForward(256, 256)
    #     ref = torch.randn(1, dim_ref, 256)  # [bs, K, dim]
    #     out = attn(memory, ref)  # [bs, h*w, dim]
    #     print(f"{memory.shape=}, {ref.shape=}, {out.shape=}")
    #     out2 = FF(out)
    #     print(f"{out2.shape=}")

    #     assert out.shape == memory.shape

    x = torch.randn(1, 330 + 8, 256)
    FF = FeedForward(256, 256)
    out = FF(x)
    print(f"{x.shape=}, {out.shape=}")
