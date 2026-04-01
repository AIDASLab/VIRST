import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_sin_cos(L: int, d_pair: int, device):
    """
    Build RoPE caches for a 1D axis.
    d_pair must be even (# of features used by RoPE on that axis).
    Returns cos,sin with shape [L, d_pair], ready to be broadcast to [B/H,...].
    """
    assert d_pair % 2 == 0
    half = d_pair // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(L, device=device, dtype=torch.float32)
    freqs = torch.einsum('l,h->lh', t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin

def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

def apply_rope_1d(q, k, cos, sin):
    """
    q,k: [..., d_pair]
    cos,sin: [L, d_pair] or broadcastable
    returns rotated q,k (same shape)
    """
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

class RoPE3D(nn.Module):
    def __init__(self, d_head: int, grid_hw: int, max_t: int):
        super().__init__()
        base = (d_head // 2) // 3 * 2
        self.d_t = base
        self.d_y = base
        self.d_x = base
        self.d_free = d_head - (self.d_t + self.d_y + self.d_x)
        self.grid = grid_hw
        self.max_t = max_t

    def build(self, T, device):
        cos_t, sin_t = build_sin_cos(T, self.d_t, device)
        cos_y, sin_y = build_sin_cos(self.grid, self.d_y, device)
        cos_x, sin_x = build_sin_cos(self.grid, self.d_x, device)
        self.register_buffer('cos_t', cos_t, persistent=False)
        self.register_buffer('sin_t', sin_t, persistent=False)
        self.register_buffer('cos_y', cos_y, persistent=False)
        self.register_buffer('sin_y', sin_y, persistent=False)
        self.register_buffer('cos_x', cos_x, persistent=False)
        self.register_buffer('sin_x', sin_x, persistent=False)

    def apply_to_memory(self, q, k, T):
        """
        Apply 3D RoPE to memory sequences laid out as S_mem = T * H * W
        q,k: [B, nH, S_mem, d_head]
        """
        if not hasattr(self, 'cos_t') or self.cos_t.size(0) < T:
            self.build(T, q.device)

        B, nH, S, d = q.shape
        H = W = self.grid
        assert S == T * H * W, f"RoPE3D expects S=T*H*W, got {S} != {T}*{H}*{W}"

        s0, s1, s2 = self.d_t, self.d_t + self.d_y, self.d_t + self.d_y + self.d_x
        qt, qy, qx, qfree = q[..., :s0], q[..., s0:s1], q[..., s1:s2], q[..., s2:]
        kt, ky, kx, kfree = k[..., :s0], k[..., s0:s1], k[..., s1:s2], k[..., s2:]

        t_idx = torch.arange(T, device=q.device).repeat_interleave(H*W)
        y_idx = torch.arange(H, device=q.device).repeat_interleave(W).repeat(T)
        x_idx = torch.arange(W, device=q.device).repeat(T*H)

        ct = self.cos_t[t_idx]
        st = self.sin_t[t_idx]
        cy = self.cos_y[y_idx]
        sy = self.sin_y[y_idx]
        cx = self.cos_x[x_idx]
        sx = self.sin_x[x_idx]

        for_name = lambda v: v.view(1,1,S,-1)
        ct, st, cy, sy, cx, sx = map(for_name, (ct,st,cy,sy,cx,sx))

        qt, kt = apply_rope_1d(qt, kt, ct, st)
        qy, ky = apply_rope_1d(qy, ky, cy, sy)
        qx, kx = apply_rope_1d(qx, kx, cx, sx)

        q = torch.cat([qt, qy, qx, qfree], dim=-1)
        k = torch.cat([kt, ky, kx, kfree], dim=-1)
        return q, k

    def apply_to_query_time(self, q, k, T):
        """
        Apply *time-only* RoPE to query sequences laid out as S_q = N * T (N objects, contiguous T).
        q,k: [B, nH, S_q, d_head]
        """
        if not hasattr(self, 'cos_t') or self.cos_t.size(0) < T:
            self.build(T, q.device)

        B, nH, S, d = q.shape
        s0, s1, s2 = self.d_t, self.d_t + self.d_y, self.d_t + self.d_y + self.d_x
        qt, qy, qx, qfree = q[..., :s0], q[..., s0:s1], q[..., s1:s2], q[..., s2:]
        kt, ky, kx, kfree = k[..., :s0], k[..., s0:s1], k[..., s1:s2], k[..., s2:]

        blocks = S // T
        t_idx = torch.arange(T, device=q.device).repeat(blocks)
        ct, st = self.cos_t[t_idx].view(1,1,S,-1), self.sin_t[t_idx].view(1,1,S,-1)

        qt, kt = apply_rope_1d(qt, kt, ct, st)
        q = torch.cat([qt, qy, qx, qfree], dim=-1)
        k = torch.cat([kt, ky, kx, kfree], dim=-1)
        return q, k
    
class RotaryMHA(nn.Module):
    def __init__(self, d_model, nhead, grid_hw=8, max_t=256, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.rope3d = RoPE3D(self.d_head, grid_hw=grid_hw, max_t=max_t)

    def forward(self, q, k, v, T_mem, T_q,
                key_padding_mask=None, attn_mask=None, return_attn=False):
        """
        q: [S_q, B, C]; k,v: [S_k, B, C]
        T_mem: temporal length in memory (S_k == T_mem*H*W)
        T_q: temporal length per object in queries (S_q == N*T_q)
        """
        S_q, B, C = q.shape
        S_k, _, _ = k.shape
        H = self.nhead
        D = self.d_head

        q = self.q_proj(q).transpose(0,1).reshape(B, S_q, H, D).transpose(1,2)
        k = self.k_proj(k).transpose(0,1).reshape(B, S_k, H, D).transpose(1,2)
        v = self.v_proj(v).transpose(0,1).reshape(B, S_k, H, D).transpose(1,2)

        q = q.contiguous(); 
        k = k.contiguous()

        q, _ = self.rope3d.apply_to_query_time(q, q.clone(), T_q)
        k, _ = self.rope3d.apply_to_memory(k, k.clone(), T_mem)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).reshape(B, S_q, H*D).transpose(0,1)
        out = self.o_proj(out)
        return (out, attn) if return_attn else out

class RoPEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, grid_hw=8, max_t=256, dropout=0.1):
        super().__init__()
        self.cross_attn = RotaryMHA(d_model, nhead, grid_hw=grid_hw, max_t=max_t, dropout=dropout)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, T_mem, T_q,
                tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attn=False):
        x = tgt
        sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout(sa)
        x = self.norm1(x)

        if return_attn:
            ca, attn = self.cross_attn(q=x, k=memory, v=memory, T_mem=T_mem, T_q=T_q,
                             key_padding_mask=memory_key_padding_mask, attn_mask=None, return_attn=return_attn)
        else:
            ca = self.cross_attn(q=x, k=memory, v=memory, T_mem=T_mem, T_q=T_q,
                             key_padding_mask=memory_key_padding_mask, attn_mask=None, return_attn=return_attn) 
        x = x + self.dropout(ca)
        x = self.norm2(x)

        y = self.ff(x)
        x = x + self.dropout(y)
        x = self.norm3(x)
        return (x, attn) if return_attn else x


class SegPrompter(nn.Module):
    def __init__(self, in_dim, token_dim=256, max_length=100, grid_hw=8, nhead=8, n_layers=2):
        super().__init__()
        self.token_dim = token_dim
        self.max_length = max_length
        self.grid_hw = grid_hw
        self.nhead = nhead
        self.n_layers = n_layers

        self.spatial_down = nn.Sequential(
            nn.Conv2d(256, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.memory_norm = nn.LayerNorm(token_dim)
        self.seg_proj = nn.Linear(in_dim, token_dim)

        self.decoder = nn.ModuleList([
            RoPEDecoderLayer(d_model=token_dim, nhead=nhead, grid_hw=grid_hw, max_t=max_length, dropout=0.1)
            for _ in range(n_layers)
        ])

    def forward(
        self, 
        seg_token, 
        image_token, 
        seg_mask = None,
        return_attn = False,
    ):
        num_conv, N, _ = seg_token.shape
        n_img, T_seg, C_in, H, W = image_token.shape
        attn_scores = []
        
        assert num_conv == n_img, f"batch mismatch: seg_token {seg_token.shape} vs image_token {image_token.shape}"
        assert T_seg <= self.max_length, f"T_seg={T_seg} exceeds max_length={self.max_length}"
        
        if seg_mask is None:
            tgt_kpm = torch.zeros((num_conv, N*T_seg), dtype=torch.bool, device=seg_token.device)
        else:
            seg_mask = seg_mask.to(torch.bool)
            tgt_keep = seg_mask.unsqueeze(2).expand(num_conv, N, T_seg).reshape(num_conv, N*T_seg)
            tgt_kpm  = ~tgt_keep

        x = image_token.reshape(num_conv * T_seg, C_in, H, W)
        x = self.spatial_down(x)
        x = x.reshape(num_conv, T_seg, self.token_dim, 64).transpose(2, 3)
        x = x.reshape(num_conv, T_seg * (self.grid_hw**2), self.token_dim).transpose(0, 1).contiguous()
        memory = self.memory_norm(x)

        seg = self.seg_proj(seg_token)
        seg = seg.unsqueeze(2).expand(num_conv, N, T_seg, self.token_dim)
        seg = seg.reshape(num_conv, N * T_seg, self.token_dim)
        seg = seg.transpose(0, 1)

        for layer in self.decoder:
            seg, attn_score = layer(
                tgt=seg,
                memory=memory,
                T_mem=T_seg,
                T_q=T_seg,
                tgt_mask=None,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=None,
                return_attn=True,
            )
            attn_scores.append(attn_score)
        decoded = seg.transpose(0, 1)
        decoded = decoded.reshape(num_conv, N, T_seg, self.token_dim)

        if return_attn:
            attn_last = attn_scores[-1]
            G2 = self.grid_hw * self.grid_hw
            attn_frame = attn_last.view(num_conv, self.nhead, N*T_seg, T_seg, G2)
            frame_score = attn_frame.mean(dim=(1,2,4))
            return decoded, frame_score
        return decoded
        
class InitialSegFusion(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 vision_dim=256, 
                 token_dim=256, 
                 grid_hw=8, 
                 num_heads=8):
        super().__init__()
        self.grid_hw = grid_hw
        self.token_dim = token_dim

        self.spatial_down = nn.Sequential(
            nn.Conv2d(vision_dim, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_norm = nn.LayerNorm(token_dim)

        self.vision_proj = nn.Linear(token_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, seg_embeds, vision_feats):
        """
        Args:
            seg_embeds: (B, N_seg, hidden_dim)
            vision_feats: (B, T_seg, C, H, W)  - raw SAM2 vision features
        Returns:
            fused_seg_embeds: (B, N_seg, hidden_dim)
        """
        B, T, C, H, W = vision_feats.shape

        x = vision_feats.flatten(0, 1)
        x = self.spatial_down(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, T, self.grid_hw**2, self.token_dim)
        x = self.spatial_norm(x)
        vision_tokens = x.reshape(B, T * self.grid_hw**2, -1)

        vision_tokens = self.vision_proj(vision_tokens)
        q = self.norm_q(seg_embeds)
        kv = self.norm_kv(vision_tokens)
        fused, _ = self.cross_attn(q, kv, kv)

        return seg_embeds + self.gate * fused
