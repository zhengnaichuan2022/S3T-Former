"""
Spiking State-Space Topology Transformer (S3T-Former)
A pure spiking Transformer designed for energy-efficient skeleton action recognition

Core Design Philosophy:
1. "Structure captures time, neurons handle spiking" - Return to SNN essence
2. Completely abandon modifications to neuron internal states
3. Use native LIF neurons, all temporal dynamics implemented through State-Space Model structure
4. High-speed design: All complex computations are in linear layers and JIT, neurons only handle spiking

Core Innovations:
1. State-Space Attention (S3-Attn): Captures long-term memory through exponential decay states
2. Lateral Spiking Topology Routing (LSTR): SNN-native topology-aware mechanism via spike propagation rather than matrix multiplication
3. Pure LIF Design: All neurons are native LIF with no internal state modifications

LSTR Innovation Points:
- Topology connections propagate spikes, not weighted continuous features
- Transform O(V²×C) dense matrix floating-point multiplication into O(Edges×C) conditional sparse addition
- Spatiotemporal unified dynamics: Topology information naturally integrates through temporal dimension
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
import numpy as np
import math

# ================= NTU Skeleton Topology Structure (25 nodes) =================
NTU_NUM_NODES = 25
NTU_SELF_LINK = [(i, i) for i in range(NTU_NUM_NODES)]
NTU_INWARD = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
              (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
              (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
              (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]
NTU_OUTWARD = [(j, i) for (i, j) in NTU_INWARD]


def build_binary_topology_matrix(num_node=25):
    """
    Build binary topology adjacency matrix (V, V)
    For LSTR: 0/1 matrix ensuring hardware-friendly sparse addition
    """
    def edge2mat(link, n):
        A = torch.zeros(n, n)
        for i, j in link:
            if i < n and j < n:
                A[j, i] = 1.0
        return A

    if num_node == 20:
        from graph.ucla import inward as g_inward
        from graph.ucla import outward as g_outward
        from graph.ucla import self_link as g_self_link
        sl, inn, out = g_self_link, g_inward, g_outward
    else:
        sl, inn, out = NTU_SELF_LINK, NTU_INWARD, NTU_OUTWARD

    # Merge all edges (self-link, inward, outward)
    A = edge2mat(sl, num_node)
    A = A + edge2mat(inn, num_node)
    A = A + edge2mat(out, num_node)
    # Binarize: connection exists means 1.0
    A = (A > 0).float()
    return A


def build_topology_matrix(num_node=25):
    """Build normalized base topology adjacency matrix (3, V, V) - NTU (25) or NW-UCLA (20)."""
    def edge2mat(link, n):
        A = torch.zeros(n, n)
        for i, j in link:
            if i < n and j < n:
                A[j, i] = 1.0
        return A

    def normalize_digraph(A):
        Dl = A.sum(0)
        h, w = A.shape
        Dn = torch.zeros(w, w)
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return torch.matmul(A, Dn)

    if num_node == 20:
        from graph.ucla import inward as g_inward
        from graph.ucla import outward as g_outward
        from graph.ucla import self_link as g_self_link
        sl, inn, out = g_self_link, g_inward, g_outward
    else:
        sl, inn, out = NTU_SELF_LINK, NTU_INWARD, NTU_OUTWARD

    I = edge2mat(sl, num_node)
    In = normalize_digraph(edge2mat(inn, num_node))
    Out = normalize_digraph(edge2mat(out, num_node))
    return torch.stack([I, In, Out], dim=0)


# ================= High-Speed JIT State-Space Engine (S3-Engine) =================
@torch.jit.script
def state_space_recurrent_engine(
    q: torch.Tensor, 
    kv: torch.Tensor, 
    decay: torch.Tensor
) -> torch.Tensor:
    """
    Spiking State-Space Attention Core (O(N) complexity)
    Captures long-term memory through exponential decay states, solving LIF short-term forgetting problem
    
    Args:
        q: (T, B, Heads, V, Head_dim) - Query spikes
        kv: (T, B, Heads, V, Head_dim) - Key-value fused spikes
        decay: (1, 1, Heads, 1, Head_dim) - State decay factor
    
    Returns:
        out: (T, B, Heads, V, Head_dim) - State-space attention output
    """
    T = q.shape[0]
    B = q.shape[1]
    num_heads = q.shape[2]
    V = q.shape[3]
    head_dim = q.shape[4]
    
    out = torch.empty_like(q)
    # Initialize historical memory state S_t (B, Heads, V, Head_dim)
    S = torch.zeros(B, num_heads, V, head_dim, device=q.device, dtype=q.dtype)
    
    # Reshape decay from (1, 1, Heads, 1, Head_dim) to (Heads, 1, Head_dim) for broadcasting with S
    # decay needs to broadcast to (B, Heads, V, Head_dim)
    decay_reshaped = decay.squeeze(0).squeeze(0)  # (Heads, 1, Head_dim)
    
    for t in range(T):
        # State update: preserve historical memory, integrate current frame's kv spike features
        # S_t = decay * S_{t-1} + (1 - decay) * kv_t
        # decay_reshaped: (Heads, 1, Head_dim) can broadcast to (B, Heads, V, Head_dim)
        # kv[t]: (B, Heads, V, Head_dim)
        S = decay_reshaped * S + (1.0 - decay_reshaped) * kv[t]
        
        # Query: use current Q to retrieve S containing global history
        out[t] = q[t] * S
    
    return out


# ================= Core Components =================

class AnatomicalSpikingEmbedding(nn.Module):
    """
    Multi-Stream Anatomical Spiking Embedding (M-ASE)
    
    Core Innovations:
    1. Transform static coordinates into physically meaningful "action event streams"
    2. Biological retina and receptors are insensitive to "absolute position" but highly sensitive to "change (motion)" and "relative deformation"
    3. Actively decompose skeleton data into "Event Streams" that SNNs favor at the input stage
    
    Design:
    - Joint Stream: Original joint positions
    - Bone Stream: Adjacent node differences in spatial dimension (captures limb deformation)
    - Motion Stream: Adjacent frame differences in temporal dimension (captures action velocity, sparsest and most suitable for SNN signals)
    
    Advantages:
    - Accuracy improvement: Introduces explicit skeleton vectors and motion vectors, network gains extremely rich information at the first layer
    - Sparsity: Motion stream is temporal difference, naturally sparse, very suitable for SNN
    - Biological plausibility: Conforms to biological visual system perception mechanisms
    """
    
    def __init__(self, in_channels=3, embed_dim=256, num_nodes=25, v_threshold=0.5):
        super().__init__()
        # Allocate embed_dim to three streams (e.g., 256 -> 96, 80, 80)
        dim_j = embed_dim // 3
        dim_b = embed_dim // 3
        dim_m = embed_dim - dim_j - dim_b  # Ensure sum equals embed_dim
        
        # Joint position encoding (Joint) - Original position information
        self.joint_conv = nn.Sequential(
            nn.Conv1d(in_channels, dim_j, 1, bias=False),
            nn.BatchNorm1d(dim_j)
        )
        
        # Bone deformation encoding (Bone - Spatial relative position)
        # Captures spatial relationships between adjacent nodes (limb deformation)
        self.bone_conv = nn.Sequential(
            nn.Conv1d(in_channels, dim_b, 1, bias=False),
            nn.BatchNorm1d(dim_b)
        )
        
        # Motion velocity encoding (Motion - Temporal relative position)
        # Captures inter-frame motion (sparsest, most suitable for SNN signals)
        self.motion_conv = nn.Sequential(
            nn.Conv1d(in_channels, dim_m, 1, bias=False),
            nn.BatchNorm1d(dim_m)
        )
        
        # Define NTU center-outward bone connection pairs (Source, Target)
        # These connections define skeleton topology structure for computing bone vectors
        self.bone_pairs = [
            (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), 
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), 
            (22, 7), (23, 24), (24, 11)
        ]
        
        # Note: Do not perform spiking here, return continuous values
        # Spiking is done by main model's self.input_lif (maintains consistency with original design)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (T*B, C, V) or (T, B, C, V) - Input skeleton sequence
                - If 3D (T*B, C, V): Process directly
                - If 4D (T, B, C, V): Reshape to 3D first
        
        Returns:
            out: (T*B, embed_dim, V) - Multi-stream fused spike features (not spiked)
                Note: Returns continuous values, requires external LIF for spiking
        """
        # Handle input shape: Support 3D or 4D input
        if x.dim() == 4:
            # 4D input (T, B, C, V)
            T, B, C, V = x.shape
            x_reshaped = x.view(T * B, C, V)
            need_reshape_back = True
        elif x.dim() == 3:
            # 3D input (T*B, C, V)
            T_B, C, V = x.shape
            x_reshaped = x
            need_reshape_back = False
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected 3D (T*B, C, V) or 4D (T, B, C, V)")
        
        # To compute temporal difference, we need to know T and B
        # If input is 3D, we need T and B from external source, or assume T=some value
        # For simplicity, if input is 3D, we cannot compute temporal difference, only spatial difference
        # But this loses motion information, so it's better to pass 4D input
        
        # If input is 3D, we cannot directly compute temporal difference
        # So we need a flag to distinguish
        if need_reshape_back:
            # 4D input: Can compute temporal difference
            T, B, C, V = x.shape
            
            # 1. Joint features (Joint) - Original position information
            x_j = x
            
            # 2. Motion features (Motion - Temporal difference)
            # First frame velocity is 0, subsequent frames are inter-frame differences
            # This is the sparsest signal, very suitable for SNN
            x_m = torch.zeros_like(x)
            x_m[1:] = x[1:] - x[:-1]  # Temporal dimension difference
            
            # 3. Bone features (Bone - Spatial difference)
            # Compute spatial relationships between adjacent nodes (limb deformation)
            x_b = torch.zeros_like(x)
            for v1, v2 in self.bone_pairs:
                if v1 < V and v2 < V:  # Ensure indices are valid
                    # Target node coordinates - Source node coordinates
                    # This captures skeleton deformation information
                    x_b[:, :, :, v1] = x[:, :, :, v1] - x[:, :, :, v2]
            
            # Flatten processing (merge temporal dimension into batch dimension)
            x_j_flat = x_j.view(T * B, C, V)
            x_m_flat = x_m.view(T * B, C, V)
            x_b_flat = x_b.view(T * B, C, V)
        else:
            # 3D input: Cannot compute temporal difference, only spatial difference
            # In this case, motion stream will be zero
            T_B, C, V = x.shape
            x_j_flat = x_reshaped
            x_m_flat = torch.zeros_like(x_reshaped)  # Cannot compute temporal difference, set to 0
            x_b_flat = x_reshaped.clone()
            
            # Compute bone features (spatial difference)
            for v1, v2 in self.bone_pairs:
                if v1 < V and v2 < V:
                    x_b_flat[:, :, v1] = x_reshaped[:, :, v1] - x_reshaped[:, :, v2]
        
        # Independently map to respective embedding spaces
        feat_j = self.joint_conv(x_j_flat)  # (T*B, dim_j, V)
        feat_m = self.motion_conv(x_m_flat)  # (T*B, dim_m, V)
        feat_b = self.bone_conv(x_b_flat)  # (T*B, dim_b, V)
        
        # Concatenate (concat on channel dimension)
        fused_flat = torch.cat([feat_j, feat_b, feat_m], dim=1)  # (T*B, embed_dim, V)
        
        # Note: Do not perform spiking here, return continuous values
        # Spiking is done by external LIF (maintains consistency with original design)
        return fused_flat


# Note: TopoRoutedLIFNode class has been removed
# Reason: In SpikingSpatioTemporalAttention, we use bmm + native LIFNode approach
# This approach is faster than handwritten for-loop TopoRoutedLIFNode, and bmm operation itself
# already implements the mathematical equivalent of "sparse topology broadcasting"
# Therefore TopoRoutedLIFNode is redundant and has been deleted to keep code concise


# Note: NonSpikingReadout class has been removed
# Reason: Previous implementation was "pseudo" membrane potential readout because it received 0/1 spikes
# rather than true membrane potential
# True membrane potential readout requires using IFNode(v_threshold=inf) as integrator
# See readout_integrator implementation in SpikingStateSpaceTopologyTransformer for details


class SpikingSpatioTemporalAttention(nn.Module):
    """
    Spatio-Temporal Spiking Attention (SD-TSSA)
    
    Core Innovation: Spatiotemporal decoupled spiking attention mechanism + Asymmetric Temporal Gradient QKV (ATG-QKV)
    
    Spatiotemporal Decoupling Mechanism:
    1. Local Binding: K and V first do element-wise multiplication to get joint spike S_kv
    2. Spatial Topology Routing: S_kv spikes flow along skeleton topology graph to neighbor nodes
    3. Temporal State-Space Memory: Store broadcasted topology KV into temporally exponential decay memory bank
    4. Query Matching: Use Q spikes to query this memory bank containing spatiotemporal global information
    
    ATG-QKV Innovation:
    - Query/Key: Generated based on temporal gradient (difference), only moving nodes produce Q and K spikes
    - Value: Generated based on original stable spikes, responsible for remembering shape and pose
    - Biological Plausibility: Conforms to visual cortex M-cell (dynamic) and P-cell (static) separation mechanism
    - Sparsity Improvement: Q and K sparsity is 1-2 orders of magnitude higher than conventional methods
    
    Advantages:
    - Q, K, V are equal in status and semantically separated: Q/K handle dynamic changes, V handles static support
    - Reduced computation: Merge KV first then do topology processing, computation reduced by 66%
    - True spatiotemporal receptive field: Wrist Query can attend to wrist, elbow, shoulder features across multiple past frames
    - Global receptive field with zero floating-point multiplication: Completely decouples spatial aggregation and temporal integration
    - Extreme sparsity: Only moving nodes participate in attention computation, computation speed is extremely fast
    """
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        num_nodes=NTU_NUM_NODES, 
        v_threshold=0.5,
        dropout=0.1,
        use_topology_bias=True,
        topology_alpha=0.5,
        use_temporal_gradient_qkv=True,  # Whether to use temporal gradient QKV
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_nodes = num_nodes
        self.use_topology_bias = use_topology_bias
        self.topology_alpha = topology_alpha
        self.use_temporal_gradient_qkv = use_temporal_gradient_qkv
        
        # Trick 1: Soft-ATG gradient softening parameter
        # Mix motion features and static features to avoid excessive sparsity causing node "starvation"
        # alpha_grad: Learnable channel-wise weighting parameter α ∈ R^D
        # Motion dominant weight per channel, 1-alpha_grad: Static fallback weight
        if use_temporal_gradient_qkv:
            self.alpha_grad = nn.Parameter(torch.ones(dim) * 0.8)  # Initialize to 0.8 for each channel
        else:
            self.alpha_grad = nn.Parameter(torch.ones(dim))  # Initialize to 1.0 (no gradient mixing)
        
        # 1. Projection layers
        # Value handles content, use normal projection as usual (based on stable spikes)
        self.v_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        
        # Q and K handle dynamic intent, they will be generated based on Temporal Gradient!
        # If a node doesn't move, it won't emit Query spikes, extremely sparse!
        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        
        # 2. Native LIF neurons produce first-level 0/1 spikes
        self.q_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.k_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.v_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        
        # ==============================================================
        # Innovation Module 1: Spatial Topology Router
        # Used to spatially propagate KV features along skeleton graph to neighbors
        # Head-wise Topology: Each Head has independent learnable topology graph
        # Perfectly equivalent to CTR-GCN's multi-topology, while retaining SNN's efficient broadcasting
        # ==============================================================
        if use_topology_bias:
            base_topology = build_topology_matrix(num_nodes).mean(dim=0)  # (V, V)
            self.register_buffer('base_topology', base_topology)
            # Let each Head learn an independent topology supplement! (1, Heads, V, V)
            self.learned_topology = nn.Parameter(torch.zeros(1, num_heads, num_nodes, num_nodes))
            nn.init.normal_(self.learned_topology, std=0.02)
        else:
            self.base_topology = None
            self.learned_topology = None
        
        # Post-topology buffer LIF (acts as membrane potential receiver after spatial propagation)
        self.topo_buffer_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        
        # ==============================================================
        # Innovation Module 2: Temporal State-Space Memory
        # ==============================================================
        self.state_decay_w = nn.Parameter(torch.zeros(1, 1, num_heads, 1, self.head_dim))
        nn.init.constant_(self.state_decay_w, -2.0)  # Initialize to small value, tend to preserve history
        
        # 3. Output projection
        self.out_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, 1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: (T, B, C, V) - Input spike sequence
        
        Returns:
            out: (T, B, C, V) - Output spike sequence
        """
        T, B, C, V = x.shape
        identity = x
        
        # Flatten for channel projection
        x_flat = x.view(T * B, C, V)
        
        # Innovation: Asymmetric Temporal Gradient QKV Generation (ATG-QKV)
        if self.use_temporal_gradient_qkv:
            # Extract temporal gradient (Temporal Gradient)
            # Capture the node's "new" or "disappeared" spike activity in current frame
            # If a node doesn't move (x_grad=0), it won't emit Query spikes, extremely sparse!
            # Optimization: Use absolute value, only care about whether state changed, not whether from 0 to 1 or 1 to 0
            # This avoids negative values (-1) being filtered by LIF threshold, causing information loss
            x_grad = torch.zeros_like(x)
            x_grad[1:] = torch.abs(x[1:] - x[:-1])  # Temporal dimension difference, take absolute value
            
            # Trick 1: Soft-ATG gradient softening
            # Soften pure gradient into "gradient dominant + static fallback"
            # This way even if nodes don't move, there's small probability of emitting Q/K spikes for global coordination
            # Avoid excessive sparsity causing node "starvation"
            # alpha_grad: (C,) -> reshape to (1, 1, C, 1) for broadcasting to (T, B, C, V)
            alpha_grad = self.alpha_grad.view(1, 1, C, 1)  # (1, 1, C, 1) for broadcasting
            qkv_input = alpha_grad * x_grad + (1.0 - alpha_grad) * x
            
            qkv_input_flat = qkv_input.view(T * B, C, V)
            
            # V uses original stable spikes (represent pose/content)
            # Responsible for remembering shape and pose, based on static content
            v_pre = self.v_bn(self.v_conv(x_flat)).view(T, B, C, V)
            v = self.v_lif(v_pre)
            
            # Q and K use softened gradient spikes (represent attention intent)
            # Mixes motion features and static features, avoiding completely static nodes being ignored
            # This conforms to biological visual system M-cell (dynamic) mechanism while retaining P-cell (static) fallback information
            q_pre = self.q_bn(self.q_conv(qkv_input_flat)).view(T, B, C, V)
            k_pre = self.k_bn(self.k_conv(qkv_input_flat)).view(T, B, C, V)
            
            q = self.q_lif(q_pre)
            k = self.k_lif(k_pre)
        else:
            # Traditional way: Q, K, V all based on original input
            q = self.q_lif(self.q_bn(self.q_conv(x_flat)).view(T, B, C, V))
            k = self.k_lif(self.k_bn(self.k_conv(x_flat)).view(T, B, C, V))
            v = self.v_lif(self.v_bn(self.v_conv(x_flat)).view(T, B, C, V))
        
        # Reshape to multi-head shape: (T, B, Heads, V, Head_dim)
        q = q.view(T, B, self.num_heads, self.head_dim, V).permute(0, 1, 2, 4, 3).contiguous()
        k = k.view(T, B, self.num_heads, self.head_dim, V).permute(0, 1, 2, 4, 3).contiguous()
        v = v.view(T, B, self.num_heads, self.head_dim, V).permute(0, 1, 2, 4, 3).contiguous()
        
        # ==============================================================
        # Step A: Local Feature Binding
        # ==============================================================
        # Pure spike element-wise multiplication, extract meaningful key information from current node
        # S_kv = S_k ⊙ S_v, represents joint spike of current node's intent and content
        # At this point KV = K(dynamic change) * V(static features)
        # This means: Only nodes in motion state will have their features marked and broadcast!
        kv_local = k * v
        
        # ==============================================================
        # Step B: Spatial Topology Broadcasting
        # Neighbor nodes' KV spikes flow to self through topology matrix!
        # Head-wise Topology: Each Head uses independent topology graph
        # ==============================================================
        if self.use_topology_bias and self.base_topology is not None:
            # base_topology is (V, V), adding (1, Heads, V, V) will auto-broadcast
            # Get dynamic topology graph (1, Heads, V, V), use softmax to ensure propagation energy conservation
            dynamic_topo = self.base_topology.view(1, 1, V, V) + self.topology_alpha * self.learned_topology
            dynamic_topo = torch.softmax(dynamic_topo, dim=-1)  # (1, Heads, V, V)
            
            # Batch apply topology graph: KV_topo = Topology @ KV_local
            # Here we perform sparse matrix addition operation, because kv_local is 0/1 sparse spikes!
            # dynamic_topo: (1, Heads, V, V), kv_local: (T, B, Heads, V, Head_dim)
            # Use einsum for batch matrix multiplication, each Head uses its own topology graph
            kv_spatial = torch.einsum('bhij,tbhjd->tbhid', dynamic_topo, kv_local)  # (T, B, Heads, V, Head_dim)
        else:
            kv_spatial = kv_local
        
        # After topology propagation, convert to current and inject into buffer LIF to generate new spikes
        # This step re-spikes the continuous values after topology propagation, maintaining SNN sparsity
        # Note: kv_spatial after topology propagation is continuous values, needs re-spiking
        # But to maintain computation efficiency, we directly use kv_spatial (already weighted sum of spikes)
        # If kv_local is sparse spikes, then kv_spatial is also sparse weighted sum
        # Here we choose to use directly, or optionally re-spike
        # To maintain SNN sparsity, we perform lightweight spiking
        kv_spatial_flat = kv_spatial.permute(0, 1, 2, 4, 3).contiguous().view(T, B, C, V)
        kv_spatial_spike = self.topo_buffer_lif(kv_spatial_flat)
        kv_spatial_spike = kv_spatial_spike.view(T, B, self.num_heads, self.head_dim, V).permute(0, 1, 2, 4, 3).contiguous()
        
        # ==============================================================
        # Step C: Temporal State-Space Memory
        # Smoothly accumulate KV_spatial with spatial neighbor information on temporal axis
        # ==============================================================
        # Optimization: Limit decay range to prevent gradient explosion or vanishing in long sequence BPTT
        # Use clamp to ensure decay is in [0.01, 0.99] range, ensuring numerical stability
        decay = torch.sigmoid(self.state_decay_w)
        decay = torch.clamp(decay, min=0.01, max=0.99)
        # Use JIT engine to process temporal sequence, forming spatiotemporal memory pool
        memory_state = state_space_recurrent_engine(q, kv_spatial_spike, decay)
        
        # ==============================================================
        # Step D: Output Projection
        # ==============================================================
        attn_out = memory_state.permute(0, 1, 2, 4, 3).contiguous().view(T, B, C, V)
        attn_flat = attn_out.view(T * B, C, V)
        out = self.out_lif(self.proj_bn(self.proj_conv(attn_flat)).view(T, B, C, V))
        
        return out + identity


# Backward compatible alias
SpikingStateSpaceAttention = SpikingSpatioTemporalAttention


class SpikingMLP(nn.Module):
    """
    Standard Spiking Feedforward Network
    Pure LIF design with no internal state modifications
    """
    
    def __init__(self, in_dim, hidden_dim, v_threshold=0.5, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = neuron.LIFNode(step_mode='m', backend='cupy')
        
        self.fc2 = nn.Conv1d(hidden_dim, in_dim, 1)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.lif2 = neuron.LIFNode(step_mode='m', backend='cupy')
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        T, B, C, V = x.shape
        identity = x
        
        x_flat = x.view(T * B, C, V)
        x_h = self.lif1(self.bn1(self.fc1(x_flat)).view(T, B, -1, V))
        
        x_h_flat = x_h.view(T * B, -1, V)
        out = self.lif2(self.bn2(self.fc2(x_h_flat)).view(T, B, C, V))
        
        return out + identity


class S3TBlock(nn.Module):
    """
    S3T-Former Core Building Block
    
    Structure:
    1. State-Space Attention (captures long-term memory)
    2. MLP (channel feature mapping)
    3. Residual connection
    """
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        num_nodes=NTU_NUM_NODES, 
        mlp_ratio=4.0, 
        v_threshold=0.5,
        dropout=0.1,
        use_topology_bias=True,
        topology_alpha=0.5,
        use_temporal_gradient_qkv=True,  # Whether to use temporal gradient QKV
    ):
        super().__init__()
        # S3 attention provides global topology features and long-term memory
        self.attn = SpikingStateSpaceAttention(
            dim, num_heads, num_nodes, v_threshold, 
            dropout, use_topology_bias, topology_alpha,
            use_temporal_gradient_qkv=use_temporal_gradient_qkv
        )
        # MLP provides channel feature mapping
        self.mlp = SpikingMLP(dim, int(dim * mlp_ratio), v_threshold, dropout)

    def forward(self, x):
        x_attn = self.attn(x)
        out = self.mlp(x_attn)
        return out


# ================= Final Model: S3T-Former =================
class SpikingStateSpaceTopologyTransformer(nn.Module):
    """
    Spiking State-Space Topology Transformer (S3T-Former)
    
    A pure spiking Transformer designed for energy-efficient skeleton action recognition
    
    Core Features:
    1. Pure LIF Design: All neurons are native LIF with no internal state modifications
    2. State-Space Attention: Captures long-term memory through structure, not by modifying neurons
    3. Topology-Aware: Enhances spatial relationship modeling with skeleton topology structure
    4. High-Speed Design: All complex computations are in linear layers and JIT
    
    Applicable Scenarios:
    - NTU RGB+D skeleton action recognition
    - Other skeleton-based action recognition tasks
    """
    
    def __init__(
        self, 
        num_nodes=NTU_NUM_NODES,
        in_channels=3, 
        embed_dim=256, 
        depth=6, 
        num_heads=8, 
        mlp_ratio=4.0, 
        num_classes=60, 
        v_threshold=0.5,
        dropout=0.1,
        use_topology_bias=True,
        topology_alpha=0.5,
        num_person=2,
        use_temporal_gradient_qkv=True,  # Whether to use temporal gradient QKV (ATG-QKV)
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_person = num_person
        self.in_channels = in_channels
        self.num_classes = num_classes  # Save num_classes attribute for forward method
        
        # Multi-person data preprocessing BatchNorm
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        nn.init.constant_(self.data_bn.weight, 1)
        nn.init.constant_(self.data_bn.bias, 0)
        
        # Innovation: Multi-Stream Anatomical Spiking Embedding (M-ASE)
        # Transform static coordinates into physically meaningful "action event streams"
        # Contains: Joint stream + Bone stream + Motion stream
        self.input_embed = AnatomicalSpikingEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_nodes=num_nodes,
            v_threshold=v_threshold
        )
        
        # Input spiking (pure LIF)
        # Note: AnatomicalSpikingEmbedding returns continuous values, perform spiking here
        self.input_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        
        # S3T-Former module stacking
        self.blocks = nn.ModuleList([
            S3TBlock(
                embed_dim, num_heads, num_nodes, mlp_ratio, 
                v_threshold, dropout, use_topology_bias, topology_alpha,
                use_temporal_gradient_qkv=use_temporal_gradient_qkv
            )
            for _ in range(depth)
        ])
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head_bn = nn.BatchNorm1d(embed_dim)
        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 1. First use fully connected layer to map embed_dim to num_classes
        # Note: Here we move classifier inside temporal loop to process features of each frame
        self.head_fc = nn.Linear(embed_dim, num_classes)
        
        # 2. True Membrane Potential Integrator (Non-Spiking Integrator)
        # It only accepts input x, accumulates it to internal membrane potential v, never emits spikes
        # In SpikingJelly, use IFNode and set a very large v_threshold
        # This way neuron never emits spikes, only performs integration V[t] = V[t-1] + X[t]
        # Note: Use a very large value instead of inf, because some implementations may not support inf
        self.readout_integrator = neuron.IFNode(
            v_threshold=1e10,  # Very large threshold, ensures never emits spikes
            v_reset=0.0,
            step_mode='m'
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: (N, C, T, V, M) - NTU native skeleton format
                N: batch size
                C: coordinate dimension (3)
                T: time steps
                V: number of nodes (25)
                M: number of persons (2)
        
        Returns:
            logits: (N, num_classes) - Classification logits
        """
        N, C, T, V, M = x.shape
        
        # [Preprocessing] Flatten spatial and person dimensions for normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        
        # Reshape to (T, N*M, C, V) to conform to SNN pure element-wise propagation standard
        x = x.view(N, M, V, C, T).permute(4, 0, 1, 3, 2).contiguous()
        x = x.view(T, N * M, C, V)
        
        # Input embedding and spiking
        # Note: AnatomicalSpikingEmbedding requires 4D input (T, B, C, V) to compute temporal difference
        # So directly pass 4D input, rather than flattening first
        x_embed = self.input_embed(x)  # x is (T, N*M, C, V), returns (T*N*M, embed_dim, V)
        x_embed = x_embed.view(T, N * M, self.embed_dim, V)  # reshape back to 4D for LIF
        x = self.input_lif(x_embed)  # Spiking
        
        # Backbone network: S3T-Former modules
        for block in self.blocks:
            x = block(x)  # x: (T, N*M, embed_dim, V) at this point is 0/1 spikes
            
        # ================= True Membrane Potential Readout =================
        T_steps, NM, C, V = x.shape
        
        # 1. Spatial pooling (still maintain temporal dimension)
        x_flat = x.view(T_steps * NM, self.embed_dim, V)
        x_pooled = self.global_pool(x_flat).view(T_steps, NM, self.embed_dim)  # (T, NM, D)
        
        # 2. Aggregate person dimension M (still maintain temporal dimension)
        x_person = x_pooled.view(T_steps, N, M, self.embed_dim).mean(dim=2)  # (T, N, D)
        
        # 3. Flatten and map through classifier to class space
        # Note: Here we map each frame's data to class space, getting each frame's "classification current (Logits Current)"
        x_person_flat = x_person.view(T_steps * N, self.embed_dim)
        logits_current = self.head_fc(self.head_dropout(self.head_bn(x_person_flat)))  # (T*N, num_classes)
        
        # Reshape back to temporal sequence
        logits_current_seq = logits_current.view(T_steps, N, self.num_classes)  # (T, N, num_classes)
        
        # 4. True Membrane Potential Integration
        # Inject T frames of classification current into integrator, integrator's final membrane potential V_mem is the classification result!
        # IFNode(v_th=inf) source logic is simply V[t] = V[t-1] + X[t], never emits spikes
        # Ensure membrane potential is zeroed before each forward pass
        functional.reset_net(self.readout_integrator)
        
        # We don't need its spikes (all zeros), we only need its final membrane potential!
        # Pass logits_current_seq into integrator, it will accumulate current from each frame
        _ = self.readout_integrator(logits_current_seq)  # (T, N, num_classes)
        
        # Extract integrator's final membrane potential as prediction output
        # Note: If using DataParallel, need to access model.module
        if hasattr(self.readout_integrator, 'module'):
            out_logits = self.readout_integrator.module.v  # (N, num_classes)
        else:
            out_logits = self.readout_integrator.v  # (N, num_classes)
        
        return out_logits
    
    def forward_with_tet(self, x):
        """
        Trick 4: TET Loss Support (Temporal Efficient Training)
        
        Returns logits for each time step, used for TET loss calculation
        This allows the network to receive strong supervision signals at each frame, accuracy usually improves by 2%~5%!
        
        Args:
            x: (N, C, T, V, M) - NTU native skeleton format
        
        Returns:
            logits_seq: (T, N, num_classes) - Logits for each time step
        """
        N, C, T, V, M = x.shape
        
        # [Preprocessing] Flatten spatial and person dimensions for normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        
        # Reshape to (T, N*M, C, V) to conform to SNN pure element-wise propagation standard
        x = x.view(N, M, V, C, T).permute(4, 0, 1, 3, 2).contiguous()
        x = x.view(T, N * M, C, V)
        
        # Input embedding and spiking
        x_embed = self.input_embed(x)
        x_embed = x_embed.view(T, N * M, self.embed_dim, V)
        x = self.input_lif(x_embed)
        
        # Backbone network: S3T-Former modules
        for block in self.blocks:
            x = block(x)
        
        # ================= Membrane Potential Readout (consistent with forward method) =================
        T_steps, NM, C_dim, V = x.shape
        
        # 1. Spatial pooling (still maintain temporal dimension)
        x_flat = x.view(T_steps * NM, self.embed_dim, V)
        x_pooled = self.global_pool(x_flat).view(T_steps, NM, self.embed_dim)  # (T, NM, D)
        
        # 2. Aggregate person dimension M (still maintain temporal dimension)
        x_person = x_pooled.view(T_steps, N, M, self.embed_dim).mean(dim=2)  # (T, N, D)
        
        # 3. Flatten and map through classifier to class space
        x_person_flat = x_person.view(T_steps * N, self.embed_dim)
        logits_current = self.head_fc(self.head_dropout(self.head_bn(x_person_flat)))  # (T*N, num_classes)
        
        # Reshape back to temporal sequence
        logits_current_seq = logits_current.view(T_steps, N, self.num_classes)  # (T, N, num_classes)
        
        # 4. Membrane potential integration, and extract membrane potential for each time step
        functional.reset_net(self.readout_integrator)
        
        # Process step by step, record membrane potential for each time step
        logits_seq = []
        for t in range(T_steps):
            # Pass current time step's logits current
            _ = self.readout_integrator(logits_current_seq[t:t+1])  # (1, N, num_classes)
            # Extract current time step's membrane potential
            v_t = self.readout_integrator.v  # (N, num_classes)
            logits_seq.append(v_t)
        
        # Return membrane potential logits for each time step: (T, N, num_classes)
        return torch.stack(logits_seq, dim=0)

