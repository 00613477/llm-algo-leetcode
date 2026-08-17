import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None,
                kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # TODO 1: split heads -> [B, H, S, D]
        xq = xq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # TODO 2: append historical KV cache along sequence dimension
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)

        # Cache compact KV heads before expansion.
        new_kv_cache = (xk, xv)

        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)

        # TODO 3: scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        # TODO 4: merge heads -> [B, S, H*D]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.o_proj(output), new_kv_cache


def test_mha_mqa_gqa():
    torch.manual_seed(42)
    batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4
    x = torch.randn(batch_size, seq_len, hidden_dim)

    print("Testing MHA (Multi-Head Attention)...")
    mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)
    out, _ = mha(x)
    assert out.shape == (batch_size, seq_len, hidden_dim), "MHA output shape error!"
    print("  ✓ MHA output shape:", tuple(out.shape))

    print("Testing GQA (Grouped-Query Attention)...")
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2)
    out, gqa_cache = gqa(x)
    assert out.shape == (batch_size, seq_len, hidden_dim), "GQA output shape error!"
    assert gqa_cache[0].shape == (batch_size, 2, seq_len, hidden_dim // num_heads)
    print("  ✓ GQA output shape:", tuple(out.shape))
    print("  ✓ Compact GQA KV shape:", tuple(gqa_cache[0].shape))

    print("Testing MQA (Multi-Query Attention)...")
    mqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=1)
    out, mqa_cache = mqa(x)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert mqa_cache[0].shape[1] == 1
    print("  ✓ MQA KV heads:", mqa_cache[0].shape[1])

    print("Testing KV Cache Autoregressive Decoding...")
    prefill_len = 5
    x_prefill = torch.randn(batch_size, prefill_len, hidden_dim)
    _, kv_cache = mha(x_prefill)
    x_decode = torch.randn(batch_size, 1, hidden_dim)
    out_decode, new_kv_cache = mha(x_decode, kv_cache=kv_cache)
    expected_cache_shape = (batch_size, num_heads, prefill_len + 1, hidden_dim // num_heads)
    assert out_decode.shape == (batch_size, 1, hidden_dim)
    assert new_kv_cache[0].shape == expected_cache_shape, "KV Cache update error!"
    print("  ✓ Decode output shape:", tuple(out_decode.shape))
    print("  ✓ Updated KV Cache shape:", tuple(new_kv_cache[0].shape))

    print("\n✅ All Tests Passed! Attention 算子实现通过测试。")


if __name__ == "__main__":
    test_mha_mqa_gqa()
