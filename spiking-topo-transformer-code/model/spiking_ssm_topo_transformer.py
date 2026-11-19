"""
S3T-Former: Spiking State-Space Topology Transformer for skeleton action recognition.

The implementation follows the method section of the paper:

* Sec. 3.1  discrete-time LIF neuron, Eq. (1):
      H[t] = tau * U[t-1] + I[t],  S[t] = Theta(H[t] - U_th),  U[t] = H[t] (1 - S[t])
  with leakage factor tau = 0.5, threshold U_th = 0.5, hard reset, and the
  arctangent (ATan) surrogate gradient with shape parameter 2.0 (Sec. 4.2).
* Sec. 3.2  M-ASE: identity, temporal-difference, and anatomical-difference
  streams, each projected by a 1D convolution + BN, concatenated, and spiked.
* Sec. 3.3  S3T Block: ATG-QKV (Soft-ATG with alpha = sigmoid(a)), binary K-V
  binding, LSTR routing A_route = Softmax_row(A_base + gamma * A_learned), a
  buffer LIF, the S3-Engine recurrence, an output LIF, and a Spiking MLP, with
  the two residual connections of Listing 1 (Appendix A).
* Sec. 3.4  global average pooling over joints and persons, a classifier that
  produces the class current I_c[t], and U-Readout U_out[t] = U_out[t-1] + I_c[t].
  The prefix logits z[t] = U_out[t] are returned for the prefix objective, Eq. (14).

Ablation switches (Sec. 4.5): use_mase, use_temporal_gradient_qkv, lstr_mode,
use_s3_engine, s3_decay_mode, and readout.
"""

import math

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

# ----------------------------------------------------------------------------
# Skeleton topologies (0-based). Inward edges (i, j) point from joint i towards
# the body centre j; M-ASE uses them as oriented anatomical edges
# (v_tgt, v_src) = (i, j), i.e. X^(S)[v_tgt] = X[v_tgt] - X[v_src].
# ----------------------------------------------------------------------------
NTU_NUM_NODES = 25
NTU_INWARD = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
              (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
              (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
              (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]

UCLA_NUM_NODES = 20
UCLA_INWARD = [(i - 1, j - 1) for (i, j) in [
    (1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    (18, 17), (19, 18), (20, 19)]]


def inward_edges(num_nodes):
    if num_nodes == NTU_NUM_NODES:
        return NTU_INWARD
    if num_nodes == UCLA_NUM_NODES:
        return UCLA_INWARD
    raise ValueError(f'No skeleton topology defined for num_nodes={num_nodes}')


def build_anatomical_prior(num_nodes):
    """A_base: average of the self-connection, normalized inward-adjacency, and
    normalized outward-adjacency matrices (Sec. 3.3.2, Step B)."""
    def edge2mat(links):
        A = torch.zeros(num_nodes, num_nodes)
        for i, j in links:
            A[j, i] = 1.0
        return A

    def normalize_digraph(A):
        deg = A.sum(0)
        inv = torch.where(deg > 0, 1.0 / deg.clamp(min=1e-12), torch.zeros_like(deg))
        return A @ torch.diag(inv)

    inward = inward_edges(num_nodes)
    outward = [(j, i) for (i, j) in inward]
    I = edge2mat([(i, i) for i in range(num_nodes)])
    A_in = normalize_digraph(edge2mat(inward))
    A_out = normalize_digraph(edge2mat(outward))
    return (I + A_in + A_out) / 3.0


# ----------------------------------------------------------------------------
# Neurons
# ----------------------------------------------------------------------------
def resolve_backend(backend='auto'):
    if backend != 'auto':
        return backend
    if torch.cuda.is_available():
        try:
            import cupy  # noqa: F401
            return 'cupy'
        except ImportError:
            pass
    return 'torch'


def make_lif(v_threshold=0.5, leak=0.5, surrogate_alpha=2.0, backend='torch'):
    """Discrete-time LIF of Eq. (1).

    SpikingJelly's LIFNode with decay_input=False and v_reset=0 computes
    H[t] = V[t-1] - V[t-1] / tau_sj + X[t], so the paper's leakage factor
    tau = 0.5 corresponds to tau_sj = 1 / (1 - tau) = 2.
    """
    return neuron.LIFNode(
        tau=1.0 / (1.0 - leak), decay_input=False, v_threshold=v_threshold,
        v_reset=0.0, detach_reset=False,
        surrogate_function=surrogate.ATan(alpha=surrogate_alpha),
        step_mode='m', backend=backend)


class ConvBN(nn.Sequential):
    """1x1 convolution followed by batch normalization, applied to (T, B, C, V)."""

    def __init__(self, c_in, c_out, bias=False):
        super().__init__(nn.Conv1d(c_in, c_out, 1, bias=bias), nn.BatchNorm1d(c_out))

    def forward(self, x):
        T, B, C, V = x.shape
        y = super().forward(x.reshape(T * B, C, V))
        return y.view(T, B, -1, V)


# ----------------------------------------------------------------------------
# S3-Engine (Step C)
# ----------------------------------------------------------------------------
@torch.jit.script
def s3_engine(q: torch.Tensor, kv: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """M[t] = lambda * M[t-1] + (1 - lambda) * KV_out[t],  O[t] = Q[t] * M[t].

    q, kv: (T, B, H, V, Dh); decay: (H, 1, Dh). M[0] = 0.
    """
    out = torch.empty_like(q)
    memory = torch.zeros_like(kv[0])
    for t in range(q.shape[0]):
        memory = decay * memory + (1.0 - decay) * kv[t]
        out[t] = q[t] * memory
    return out


# ----------------------------------------------------------------------------
# M-ASE (Sec. 3.2)
# ----------------------------------------------------------------------------
class AnatomicalSpikingEmbedding(nn.Module):
    """Multi-Stream Anatomical Spiking Embedding.

    X^(0)[t] = X[t];  X^(T)[t] = X[t] - X[t-1] with X^(T)[0] = 0;
    X^(S)[t, :, v_tgt] = X[t, :, v_tgt] - X[t, :, v_src] for each oriented
    anatomical edge. Each stream is projected by F (1D conv + BN); the
    concatenation is spiked by the input LIF of the main model (Eq. 3).
    use_mase=False keeps a single joint projection (ablation "w/o M-ASE").
    """

    def __init__(self, in_channels=3, embed_dim=256, num_nodes=25, use_mase=True):
        super().__init__()
        self.use_mase = bool(use_mase)
        if not self.use_mase:
            self.proj_0 = ConvBN(in_channels, embed_dim)
            return
        d0 = embed_dim // 3
        ds = embed_dim // 3
        dt = embed_dim - d0 - ds
        self.proj_0 = ConvBN(in_channels, d0)
        self.proj_s = ConvBN(in_channels, ds)
        self.proj_t = ConvBN(in_channels, dt)
        edges = inward_edges(num_nodes)
        self.register_buffer('tgt', torch.tensor([i for i, _ in edges], dtype=torch.long), persistent=False)
        self.register_buffer('src', torch.tensor([j for _, j in edges], dtype=torch.long), persistent=False)

    def forward(self, x):
        """x: (T, B, C, V) -> (T, B, D, V) continuous currents."""
        if not self.use_mase:
            return self.proj_0(x)
        x_t = torch.zeros_like(x)
        x_t[1:] = x[1:] - x[:-1]
        x_s = torch.zeros_like(x)
        x_s[..., self.tgt] = x[..., self.tgt] - x[..., self.src]
        return torch.cat([self.proj_0(x), self.proj_s(x_s), self.proj_t(x_t)], dim=2)


# ----------------------------------------------------------------------------
# S3T attention: ATG-QKV + local binding + LSTR + S3-Engine (Sec. 3.3)
# ----------------------------------------------------------------------------
class S3TAttention(nn.Module):
    LSTR_MODES = ('hybrid', 'learned', 'physical', 'identity')

    def __init__(self, dim, num_heads=8, num_nodes=25, v_threshold=0.5, leak=0.5,
                 surrogate_alpha=2.0, backend='torch', topology_gamma=0.5,
                 use_temporal_gradient_qkv=True, atg_alpha_init=0.8,
                 lstr_mode='hybrid', use_s3_engine=True, s3_decay_mode='learnable',
                 s3_fixed_decay=0.5, s3_decay_init=-2.0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by num_heads {num_heads}'
        assert lstr_mode in self.LSTR_MODES, f'lstr_mode must be one of {self.LSTR_MODES}'
        assert s3_decay_mode in ('learnable', 'fixed')
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.topology_gamma = topology_gamma
        self.use_temporal_gradient_qkv = bool(use_temporal_gradient_qkv)
        self.lstr_mode = lstr_mode
        self.use_s3_engine = bool(use_s3_engine)
        self.s3_decay_mode = s3_decay_mode
        self.s3_fixed_decay = float(s3_fixed_decay)
        lif = dict(v_threshold=v_threshold, leak=leak, surrogate_alpha=surrogate_alpha, backend=backend)

        # ATG-QKV: alpha = sigmoid(a) in (0, 1)^D, channel-wise (Eq. 4).
        if self.use_temporal_gradient_qkv:
            init = math.log(atg_alpha_init / (1.0 - atg_alpha_init))
            self.atg_logit = nn.Parameter(torch.full((dim,), init))
        self.proj_q = ConvBN(dim, dim)
        self.proj_k = ConvBN(dim, dim)
        self.proj_v = ConvBN(dim, dim)
        self.lif_q, self.lif_k, self.lif_v = make_lif(**lif), make_lif(**lif), make_lif(**lif)

        # LSTR (Eq. 8): A_route^(h) = Softmax_row(A_base + gamma * A_learned^(h)).
        if lstr_mode in ('hybrid', 'physical'):
            self.register_buffer('base_topology', build_anatomical_prior(num_nodes))
        else:
            self.base_topology = None
        if lstr_mode in ('hybrid', 'learned'):
            # Head-specific residual initialized near zero.
            self.learned_topology = nn.Parameter(torch.randn(num_heads, num_nodes, num_nodes) * 0.02)
        else:
            self.learned_topology = None
        self.lif_buffer = make_lif(**lif)

        # S3-Engine: lambda = Clamp(sigmoid(w_decay), 0.01, 0.99), one value per channel.
        self.decay_logit = nn.Parameter(torch.full((num_heads, 1, self.head_dim), float(s3_decay_init)),
                                        requires_grad=self.use_s3_engine and s3_decay_mode == 'learnable')

        self.proj_out = ConvBN(dim, dim, bias=True)
        self.lif_out = make_lif(**lif)

    # -- helpers -------------------------------------------------------------
    def atg_alpha(self):
        return torch.sigmoid(self.atg_logit)

    def routing_matrix(self):
        """Row-normalized routing matrices, shape (H, V, V); None for identity routing."""
        if self.lstr_mode == 'identity':
            return None
        if self.lstr_mode == 'hybrid':
            logits = self.base_topology.unsqueeze(0) + self.topology_gamma * self.learned_topology
        elif self.lstr_mode == 'learned':
            logits = self.learned_topology
        else:  # physical
            logits = self.base_topology.unsqueeze(0).expand(self.num_heads, -1, -1)
        return torch.softmax(logits, dim=-1)

    def decay(self):
        if self.s3_decay_mode == 'fixed':
            lam = torch.full_like(self.decay_logit, self.s3_fixed_decay)
        else:
            lam = torch.sigmoid(self.decay_logit)
        return torch.clamp(lam, min=0.01, max=0.99)

    def _split(self, x):  # (T, B, D, V) -> (T, B, H, V, Dh)
        T, B, D, V = x.shape
        return x.view(T, B, self.num_heads, self.head_dim, V).permute(0, 1, 2, 4, 3).contiguous()

    def _merge(self, x):  # (T, B, H, V, Dh) -> (T, B, D, V)
        T, B, H, V, Dh = x.shape
        return x.permute(0, 1, 2, 4, 3).contiguous().view(T, B, H * Dh, V)

    # -- forward -------------------------------------------------------------
    def forward(self, x):
        """x: (T, B, D, V) binary spikes S^(l-1). Returns LIF_Out(Proj_Out(O)) (before the residual)."""
        if self.use_temporal_gradient_qkv:
            s_grad = torch.zeros_like(x)
            s_grad[1:] = torch.abs(x[1:] - x[:-1])
            alpha = self.atg_alpha().view(1, 1, -1, 1)
            s_dyn = alpha * s_grad + (1.0 - alpha) * x  # Eq. (4)
        else:
            s_dyn = x  # ablation: direct Q/K/V projection
        q = self.lif_q(self.proj_q(s_dyn))
        k = self.lif_k(self.proj_k(s_dyn))
        v = self.lif_v(self.proj_v(x))
        q, k, v = self._split(q), self._split(k), self._split(v)

        kv_local = k * v  # Step A: binary local binding (Eq. 7)

        route = self.routing_matrix()  # Step B: LSTR (Eqs. 8-10)
        kv_spatial = kv_local if route is None else torch.einsum('hij,tbhjd->tbhid', route, kv_local)
        kv_out = self._split(self.lif_buffer(self._merge(kv_spatial)))

        if self.use_s3_engine:  # Step C: S3-Engine (Eqs. 11-12)
            out = s3_engine(q, kv_out, self.decay())
        else:  # ablation: O[t] = Q[t] * KV_out[t]
            out = q * kv_out
        return self.lif_out(self.proj_out(self._merge(out)))


class SpikingMLP(nn.Module):
    """Two 1x1 projections, each followed by BN and an LIF neuron (M1 and M2 in Table 2)."""

    def __init__(self, dim, hidden_dim, **lif):
        super().__init__()
        self.fc1, self.lif1 = ConvBN(dim, hidden_dim, bias=True), make_lif(**lif)
        self.fc2, self.lif2 = ConvBN(hidden_dim, dim, bias=True), make_lif(**lif)

    def forward(self, x):
        return self.lif2(self.fc2(self.lif1(self.fc1(x))))


class S3TBlock(nn.Module):
    """S3T Block of Listing 1: x = LIF_Out(Proj_Out(S3T-Attn(x))) + x;
    return LIF_Residual(SpikingMLP(x) + x)."""

    def __init__(self, dim, num_heads=8, num_nodes=25, mlp_ratio=4.0, v_threshold=0.5, leak=0.5,
                 surrogate_alpha=2.0, backend='torch', **attn_kwargs):
        super().__init__()
        lif = dict(v_threshold=v_threshold, leak=leak, surrogate_alpha=surrogate_alpha, backend=backend)
        self.attn = S3TAttention(dim, num_heads, num_nodes, **lif, **attn_kwargs)
        self.mlp = SpikingMLP(dim, int(dim * mlp_ratio), **lif)
        self.lif_residual = make_lif(**lif)

    def forward(self, x):
        x = self.attn(x) + x
        return self.lif_residual(self.mlp(x) + x)


# ----------------------------------------------------------------------------
# S3T-Former (Sec. 3.4)
# ----------------------------------------------------------------------------
class SpikingStateSpaceTopologyTransformer(nn.Module):
    READOUTS = ('u_readout', 'spike_count')

    def __init__(self, num_nodes=25, in_channels=3, embed_dim=256, depth=6, num_heads=8,
                 mlp_ratio=4.0, num_classes=60, num_person=2, dropout=0.0,
                 v_threshold=0.5, leak=0.5, surrogate_alpha=2.0, backend='auto',
                 topology_gamma=0.5, use_mase=True, use_temporal_gradient_qkv=True,
                 atg_alpha_init=0.8, lstr_mode='hybrid', use_s3_engine=True,
                 s3_decay_mode='learnable', s3_fixed_decay=0.5, s3_decay_init=-2.0,
                 readout='u_readout'):
        super().__init__()
        assert readout in self.READOUTS, f'readout must be one of {self.READOUTS}'
        backend = resolve_backend(backend)
        lif = dict(v_threshold=v_threshold, leak=leak, surrogate_alpha=surrogate_alpha, backend=backend)
        self.num_classes, self.embed_dim, self.readout = num_classes, embed_dim, readout

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        self.input_embed = AnatomicalSpikingEmbedding(in_channels, embed_dim, num_nodes, use_mase)
        self.lif_in = make_lif(**lif)
        self.blocks = nn.ModuleList([
            S3TBlock(embed_dim, num_heads, num_nodes, mlp_ratio, **lif,
                     topology_gamma=topology_gamma,
                     use_temporal_gradient_qkv=use_temporal_gradient_qkv,
                     atg_alpha_init=atg_alpha_init, lstr_mode=lstr_mode,
                     use_s3_engine=use_s3_engine, s3_decay_mode=s3_decay_mode,
                     s3_fixed_decay=s3_fixed_decay, s3_decay_init=s3_decay_init)
            for _ in range(depth)])
        self.head_bn = nn.BatchNorm1d(embed_dim)
        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head_fc = nn.Linear(embed_dim, num_classes)
        if readout == 'spike_count':
            self.lif_readout = make_lif(**lif)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def class_current(self, x):
        """x: (N, C, T, V, M) -> class current I_c[t], shape (T, N, num_classes)."""
        N, C, T, V, M = x.shape
        x = self.data_bn(x.permute(0, 4, 3, 1, 2).reshape(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(4, 0, 1, 3, 2).reshape(T, N * M, C, V)
        s = self.lif_in(self.input_embed(x))  # S_in (Eq. 3)
        for block in self.blocks:
            s = block(s)
        g = s.mean(dim=3).view(T, N, M, self.embed_dim).mean(dim=2)  # GAP over joints and persons
        current = self.head_fc(self.head_dropout(self.head_bn(g.reshape(T * N, -1))))
        return current.view(T, N, self.num_classes)

    def forward_prefix(self, x):
        """Prefix logits z[t] for t = 1..T, shape (T, N, num_classes).

        U-Readout (Eq. 13): z[t] = U_out[t] = sum_{k<=t} I_c[k], a non-spiking,
        non-reset integrator. Spike-count readout (ablation): output LIF neurons
        spike on I_c[t] and z[t] is the spike count up to step t.
        """
        current = self.class_current(x)
        if self.readout == 'u_readout':
            return torch.cumsum(current, dim=0)
        return torch.cumsum(self.lif_readout(current), dim=0)

    def forward(self, x, return_prefix=False):
        """Returns z[T] (used for inference), or with return_prefix=True all prefix
        logits in batch-first layout (N, T, num_classes) so that DataParallel
        gathers them along the batch dimension."""
        z = self.forward_prefix(x)
        return z.transpose(0, 1) if return_prefix else z[-1]


def prefix_loss(prefix_logits, target):
    """Eq. (14): L_prefix = (1/T) sum_t CE(z[t], y)."""
    T = prefix_logits.shape[0]
    return nn.functional.cross_entropy(prefix_logits.flatten(0, 1), target.repeat(T))


__all__ = ['SpikingStateSpaceTopologyTransformer', 'prefix_loss']
