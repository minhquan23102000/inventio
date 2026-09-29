"""The gated delta rule of the hybrid (Qwen3.5) DeltaNet layers, chunked for Apple GPUs.

On MPS there is no compiled kernel for these layers (flash-linear-attention is Triton, CUDA only), so
transformers runs its reference `torch_chunk_gated_delta_rule`, and 18 of v5's 24 layers spend most of a
query there. This is the same algorithm (WY form, chunks of 64) with two changes that keep the work in
large batched matmuls: the per-chunk triangular inverse is built by block doubling (log2(64) = 6 batched
steps) where the reference walks 64 rows one at a time, and the sequential part carries only the
[Dk, Dv] state from chunk to chunk.

Measured on an M3 (inputs one real layer received at 3,000 tokens): 0.043 s against the reference's
0.144 s, 0.123 s against 0.428 s at 9,000; outputs within 5e-4 of the reference, final state within 5e-6.
Real queries with links on went from 28.6 s to 17.5 s on average, with the same top three
(benchmarks/systemone.py backend measures the pass).
"""
import torch
import torch.nn.functional as F

CHUNK = 64   # the size measured; the reference's own default


def _unit_lower_inverse(a):
    """(I + a)^-1 for `a` strictly lower triangular [..., C, C], C a power of two, by block doubling:
    L = [[P, 0], [Q, R]] has L^-1 = [[P^-1, 0], [-R^-1 Q P^-1, R^-1]]. Every intermediate is an entry of a
    true inverse; a Neumann product (I - a)(I + a^2)(I + a^4)... is shorter but overflows to inf on
    real states (a^32 before it cancels), which turned a whole pass into NaN."""
    c = a.shape[-1]
    lead = a.shape[:-2]
    inv = torch.ones(*lead, c, 1, 1, dtype=a.dtype, device=a.device)
    s = 1
    while s < c:
        nb = c // (2 * s)
        blk = inv.reshape(*lead, nb, 2, s, s)
        p_inv, r_inv = blk[..., 0, :, :], blk[..., 1, :, :]
        q = a.reshape(*lead, nb, 2 * s, nb, 2 * s).diagonal(dim1=-4, dim2=-2).movedim(-1, -3)[..., s:, :s]
        top = torch.cat([p_inv, torch.zeros_like(p_inv)], -1)
        bottom = torch.cat([-(r_inv @ q @ p_inv), r_inv], -1)
        inv = torch.cat([top, bottom], -2)
        s *= 2
    return inv.reshape(*lead, c, c)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=CHUNK, initial_state=None,
                           output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    """Same arguments and return value as transformers' `torch_chunk_gated_delta_rule`."""
    dtype = query.dtype
    b_, t_, _, dk = key.shape
    h, dv = value.shape[-2:]
    c = CHUNK
    q, k, v, beta, g = (x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * dk ** -0.5
    pad = (c - t_ % c) % c
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n = (t_ + pad) // c
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(b_, h, n, c, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    cum = g.reshape(b_, h, n, c).cumsum(-1)
    upper = torch.ones(c, c, dtype=torch.bool, device=q.device).triu(1)
    decay = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
    lower = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(~torch.ones_like(upper).tril(-1), 0)
    inv = _unit_lower_inverse(lower)
    new_values = inv @ v_beta
    k_cumdecay = inv @ (k_beta * cum.exp()[..., None])
    intra = (q @ k.transpose(-1, -2)) * decay
    q_decayed = q * cum.exp()[..., None]
    k_decayed = k * (cum[..., -1:] - cum).exp()[..., None]
    chunk_decay = cum[..., -1].exp()[..., None, None]
    state = (torch.zeros(b_, h, dk, dv, dtype=torch.float32, device=q.device) if initial_state is None
             else initial_state.to(torch.float32))
    out = []
    for i in range(n):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out.append(q_decayed[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + k_decayed[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(out, 2).reshape(b_, h, -1, dv)[:, :, :t_].transpose(1, 2).to(dtype)
    return out, (state if output_final_state else None)


def install():
    """Use this rule in place of transformers' reference one (MPS only: CUDA has flash-linear-attention)."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    qwen.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
