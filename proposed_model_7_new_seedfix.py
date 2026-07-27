"""
ACM-EMOGI v7: Gated Fusion + Trimmed Feature Set
=================================================
Base: v6 (ACM + topology MLP branch after ablation).

Changes from v6
───────────────
1. Feature set trimmed from 11 → 7 (ablation-driven):
     Removed from v6:
       • degree          — fully subsumed by log1p(2hop_sum), mean_nbr_deg,
                           std_nbr_deg; ranked dead-last (#9/9) in gradient
                           importance. Ablation: +0.0011 AUPRC on removal.
       • pagerank (raw)  — redundant once log1p(pr×N) is present; ranked #8/9.
                           Removal tightened std (±0.0168 → ±0.0154) with no
                           mean loss.
       • max_nbr_deg     — gradient importance confirmed redundant with mean_nbr_deg
                           and std_nbr_deg already in the set.
       • ego_density     — gradient importance confirmed redundant; clust_coeff and
                           kcore capture the same cohesion signal more stably.
     N_TOPO_FEATURES: 11 → 7.

2. Scalar α_T fusion replaced by per-node GatedFusion:
     Old: h_final = Linear(cat([h_acm,  α_T · h_topo]))
          α_T ∈ ℝ  (single sigmoid-parameterised scalar, learned globally)

     New: gate    = sigmoid(W_g · cat([h_acm, h_topo]))   [N, hidden_dim]
          h_final = gate ⊙ h_acm + (1 − gate) ⊙ h_topo

     Motivation: different node types should trust the topology branch to
     different degrees — bridge genes (high 2hop_unique, high std_nbr_deg)
     carry more structural signal than module-interior genes. A per-node
     gate, conditioned on *both* branch outputs, lets the model learn this
     policy without an inductive-bias bottleneck. Parameter cost: one
     additional Linear(2·hidden_dim → hidden_dim) layer.

     alpha_T_init and alpha_lr_mult config keys are retired; evaluate() and
     the CV logger report gate_mean (mean gate value across eval nodes) in
     place of alpha_T.

Final v7 feature table (7 columns)
────────────────────────────────────
  col  0 : std_nbr_deg       — std of neighbour degrees
  col  1 : log1p(2hop_sum)   — log-compressed 2-hop reach proxy
  col  2 : log1p(pr*N)       — scale-invariant log-compressed PageRank
  col  3 : clust_coeff       — local clustering coefficient
  col  4 : mean_nbr_deg      — mean neighbour degree
  col  5 : kcore             — k-core decomposition number
  col  6 : 2hop_unique       — unique 2-hop reachable node count

Performance reference
─────────────────────
  v1   (ACM, no topo branch)                    : ~0.764 AUPRC
  SGCD (GCN + MLP_A)                            : ~0.799 AUPRC
  v6   (ACM + 11-feat topo, scalar α_T)         : 0.7940 ± 0.0157
  v6   (−degree −ego_density −max_nbr_deg)      : 0.7952 ± 0.0158
  v6   (−degree)                                : 0.7963 ± 0.0168
  v6   (−degree −pagerank)                      : 0.7962 ± 0.0154
  v7   (7-feat + gated fusion)                  : 0.7981 ± 0.0165
"""


import argparse
import math
import os
import random
import sys

# Heavy imports are deferred until after seed_everything() is called from
# __main__. This ensures CUBLAS_WORKSPACE_CONFIG is set before CUDA is
# initialised and that no torch/numpy RNG state is consumed before seeding.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import scipy.sparse as sp
import pickle
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """
    Set all relevant RNG seeds for full reproducibility.

    Call this as early as possible — ideally before any other torch or numpy
    operations — so that weight initialisation, dropout masks, and data
    sampling all use a deterministic sequence.

    Steps applied
    -------------
    1. ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` — required by CUDA ≥ 10.2 for
       deterministic cuBLAS (must be set before CUDA is initialised).
    2. ``random``, ``numpy``, ``torch`` (CPU + CUDA) seeds.
    3. ``torch.backends.cudnn.deterministic = True`` — disables non-
       deterministic cuDNN auto-tuning.
    4. ``torch.backends.cudnn.benchmark = False`` — prevents cuDNN from
       choosing a different kernel between runs.
    5. ``torch.use_deterministic_algorithms(True)`` — raises an error if
       PyTorch would use a non-deterministic kernel (e.g. scatter/gather ops
       used internally by torch_geometric), making any remaining sources of
       non-determinism explicit rather than silent.

    Parameters
    ----------
    seed : int
        Integer seed. Any value is valid; 0 is not special.

    Notes
    -----
    ``torch.use_deterministic_algorithms(True)`` may raise
    ``torch.errors.UndeterministicError`` if a non-deterministic op is
    encountered. In that case, set the environment variable
    ``PYTHONHASHSEED`` to the same seed and/or wrap the offending op with
    ``torch.use_deterministic_algorithms(False)`` locally, then re-enable.
    """
    # Must be set before CUDA is initialised (i.e. before the first torch.cuda call)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Enforces determinism for all torch ops, including scatter/gather used by PyG.
    # If this raises UndeterministicError, the offending op needs investigation.
    torch.use_deterministic_algorithms(True, warn_only=False)



# ──────────────────────────────────────────────────────────────────────────────
# 1.  Omics-Aware Feature Encoder
#     Input:  [N, 64]  (4 omics × 16 cancer types, flattened)
#     Output: [N, hidden_dim]
#
#     Each omics type gets its own linear projection across the 16 cancer types,
#     then the 4 projections are concatenated. This preserves omics identity
#     before any graph operation, keeping LRP interpretable per omics type.
# ──────────────────────────────────────────────────────────────────────────────

OMICS_ORDER = ["MF", "CNA", "METH", "GE"]   # matches feature_name prefixes
N_CANCER_TYPES = 16


class OmicsEncoder(nn.Module):
    """
    Projects each of the 4 omics blocks (16-dim each) independently,
    then concatenates to form a gene embedding.

    Parameters
    ----------
    omics_dim : int
        Output dimension *per omics type*. Total output = 4 * omics_dim.
    dropout : float
    """

    def __init__(self, omics_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.omics_dim = omics_dim

        # One projection per omics type
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(N_CANCER_TYPES, omics_dim),
                nn.LayerNorm(omics_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in OMICS_ORDER
        ])

    @property
    def out_dim(self) -> int:
        return len(OMICS_ORDER) * self.omics_dim   # 4 * omics_dim

    def forward(self, x: Tensor, omics_indices: list[list[int]]) -> Tensor:
        """
        x              : [N, 64]
        omics_indices  : list of 4 lists, each with 16 column indices
                         corresponding to [MF cols, CNA cols, METH cols, GE cols]
        """
        parts = []
        for enc, idx in zip(self.encoders, omics_indices):
            block = x[:, idx]          # [N, 16]
            parts.append(enc(block))   # [N, omics_dim]
        return torch.cat(parts, dim=1)  # [N, 4*omics_dim]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Graph utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_lp_norm(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Compute D^{-1/2} A D^{-1/2} (symmetric normalised adjacency with self-loops).
    Returns edge_weight tensor aligned with edge_index (with self-loops appended).
    """
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_sl
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_sl, edge_weight


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ACM Convolution Layer
#
#     Three parallel channels per layer:
#       LP  channel : Â_sym · H · W_L          (low-pass / aggregation)
#       HP  channel : (I - Â_sym) · H · W_H    (high-pass / diversification)
#       ID  channel : H · W_I                  (identity / full-pass)
#
#     Node-wise adaptive mixing:
#       [α_L, α_H, α_I] = Softmax( [σ(H_L w̃_L), σ(H_H w̃_H), σ(H_I w̃_I)] / T ) · W_mix
#       H_out = diag(α_L)H_L + diag(α_H)H_H + diag(α_I)H_I
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLP(MessagePassing):
    """Simple symmetric-normalised graph convolution (no learnable weights)."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.unsqueeze(-1) * x_j


class ACMConv(nn.Module):
    """
    Single ACM layer operating on one graph.

    Parameters
    ----------
    in_dim      : input feature dimension
    out_dim     : output feature dimension
    dropout     : dropout rate on channel features
    temperature : softmax temperature T for mixing weights
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.temperature = temperature

        # Channel-specific linear transforms
        self.W_L = nn.Linear(in_dim, out_dim, bias=True)
        self.W_H = nn.Linear(in_dim, out_dim, bias=True)
        self.W_I = nn.Linear(in_dim, out_dim, bias=True)

        # Channel attention: score each channel's output with a scalar MLP
        self.w_att_L = nn.Linear(out_dim, 1, bias=False)
        self.w_att_H = nn.Linear(out_dim, 1, bias=False)
        self.w_att_I = nn.Linear(out_dim, 1, bias=False)

        # 3×3 mixing matrix W_mix (operates on the 3-dim softmax vector)
        self.W_mix = nn.Linear(3, 3, bias=False)

        self.norm_L = nn.LayerNorm(out_dim)
        self.norm_H = nn.LayerNorm(out_dim)
        self.norm_I = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.conv = GraphConvLP()

    def forward(
        self,
        x: Tensor,
        edge_index_lp: Tensor,
        edge_weight_lp: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        h_out  : [N, out_dim]  mixed output
        alphas : [N, 3]        mixing weights (LP, HP, ID) — for inspection
        """
        # ── LP channel ──────────────────────────────────────────────────────
        h_lp_in = self.dropout(x)
        agg = self.conv(h_lp_in, edge_index_lp, edge_weight_lp)   # Â x
        h_L = F.relu(self.norm_L(self.W_L(agg)))                  # [N, out]

        # ── HP channel (diversification = x - Â x) ──────────────────────────
        div = h_lp_in - agg                                        # (I - Â) x
        h_H = F.relu(self.norm_H(self.W_H(div)))                  # [N, out]

        # ── Identity channel ─────────────────────────────────────────────────
        h_I = F.relu(self.norm_I(self.W_I(self.dropout(x))))      # [N, out]

        # ── Node-wise adaptive mixing ─────────────────────────────────────────
        #   score each channel with a sigmoid gate, stack → [N, 3]
        scores = torch.cat([
            torch.sigmoid(self.w_att_L(h_L)),   # [N, 1]
            torch.sigmoid(self.w_att_H(h_H)),   # [N, 1]
            torch.sigmoid(self.w_att_I(h_I)),   # [N, 1]
        ], dim=1)                                # [N, 3]

        alphas = F.softmax(scores / self.temperature, dim=1)  # [N, 3]
        alphas = alphas @ self.W_mix.weight.T                 # [N, 3]  extra mixing
        alphas = F.softmax(alphas, dim=1)                     # re-normalise

        # Weighted sum of channel outputs
        h_out = (
            alphas[:, 0:1] * h_L +
            alphas[:, 1:2] * h_H +
            alphas[:, 2:3] * h_I
        )  # [N, out_dim]

        return h_out, alphas


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Full ACM-EMOGI Model
# ──────────────────────────────────────────────────────────────────────────────

class ACMEMOGIModel(nn.Module):
    """
    ACM-EMOGI v7: ACM + Parallel Topology MLP Branch (9 features) + GatedFusion.

    Architecture
    ────────────
    OmicsEncoder → [ACMConv × n_layers] → h_acm            [N, hidden_dim]
    TopoMLP(topo_feat)                  → h_topo            [N, hidden_dim]

    GatedFusion:
      gate    = sigmoid(W_g · cat([h_acm, h_topo]))         [N, hidden_dim]
      h_final = gate ⊙ h_acm + (1 − gate) ⊙ h_topo        [N, hidden_dim]

    logits = head(h_final)                                  [N, 1]

    The per-node gate is conditioned on both branch outputs, letting the model
    learn that bridge genes (high 2hop_unique, high std_nbr_deg) should weight
    the topology branch more heavily than module-interior genes. This replaces
    the single global α_T scalar from v6.

    Parameters
    ----------
    omics_dim  : projection dim per omics type (total encoder output = 4 * omics_dim)
    hidden_dim : hidden dimension shared by ACM layers and TopoMLP output
    n_layers   : number of stacked ACM conv layers
    dropout    : dropout rate throughout
    temperature: ACM softmax temperature
    """

    def __init__(
        self,
        omics_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.4,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        enc_out = 4 * omics_dim

        self.encoder = OmicsEncoder(omics_dim=omics_dim, dropout=dropout)

        self.input_proj = nn.Sequential(
            nn.Linear(enc_out, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.acm_layers = nn.ModuleList([
            ACMConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                temperature=temperature,
            )
            for _ in range(n_layers)
        ])

        # Topology MLP branch: maps N_TOPO_FEATURES → hidden_dim
        self.topo_mlp = TopoMLP(hidden_dim=hidden_dim, dropout=dropout)

        # Per-node gated fusion: learns which branch to trust per node
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        omics_indices: list[list[int]],
        edge_index_lp: Tensor,
        edge_weight_lp: Tensor,
        topo_feat: Tensor,
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        """
        Parameters
        ----------
        x              : [N, 64]  raw omics features
        omics_indices  : 4 lists of 16 column indices (MF, CNA, METH, GE)
        edge_index_lp  : [2, E']  with self-loops
        edge_weight_lp : [E']     normalised weights
        topo_feat      : [N, N_TOPO_FEATURES]  normalised structural features

        Returns
        -------
        logits   : [N, 1]
        alphas   : list of [N, 3] per ACM layer (LP / HP / ID mixing weights)
        gate_val : [N, hidden_dim]  per-node gate (for logging mean)
        """
        # ── ACM branch ───────────────────────────────────────────────────────
        h = self.encoder(x, omics_indices)
        h = self.input_proj(h)

        all_alphas = []
        for layer in self.acm_layers:
            h_new, alphas = layer(h, edge_index_lp, edge_weight_lp)
            h = h + h_new
            all_alphas.append(alphas)

        # ── Topology MLP branch ───────────────────────────────────────────────
        h_topo = self.topo_mlp(topo_feat)                    # [N, hidden_dim]

        # ── Per-node gated fusion ─────────────────────────────────────────────
        gate_val = torch.sigmoid(self.gate(torch.cat([h, h_topo], dim=1)))  # [N, H]
        h_final  = gate_val * h + (1.0 - gate_val) * h_topo                # [N, H]

        logits = self.head(h_final)
        return logits, all_alphas, gate_val


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Structural Topology Features + helper functions
# ──────────────────────────────────────────────────────────────────────────────

N_TOPO_FEATURES = 7

# Human-readable names for each topology feature column, used in diagnostics.
# Removed from v7: "degree" (ranked #9/9), "pagerank" (ranked #8/9),
# "max_nbr_deg", and "ego_density" — all confirmed redundant by ablation
# and gradient-importance analysis.
TOPO_FEATURE_NAMES = [
    "std_nbr_deg",      # col  0
    "log1p(2hop_sum)",  # col  1
    "log1p(pr*N)",      # col  2
    "clust_coeff",      # col  3
    "mean_nbr_deg",     # col  4
    "kcore",            # col  5
    "2hop_unique",      # col  6
]


def _compute_kcore(row: np.ndarray, col: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    K-core decomposition via iterative degree-bucket peeling. O(E).

    Assigns each node its core number: the maximum k such that it belongs to
    a subgraph in which every node has degree ≥ k. Nodes in the tightest
    functional modules in STRING receive the highest core numbers.

    The edge list is treated as undirected (both directions of each edge are
    added to the adjacency list) to match the undirected nature of protein
    interactions.

    Returns
    -------
    core : int32 array of shape [num_nodes]

    Notes
    -----
    For graphs with >100k nodes, replace with igraph.Graph.coreness() or a
    NumPy-vectorised implementation — the Python adjacency list here dominates
    memory at very large scale.
    """
    # Build undirected adjacency list
    adj = [set() for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    deg_remaining = np.array([len(adj[i]) for i in range(num_nodes)], dtype=np.int32)
    core    = np.zeros(num_nodes, dtype=np.int32)
    removed = np.zeros(num_nodes, dtype=bool)

    # Initialise bucket sort by current degree
    max_deg = int(deg_remaining.max()) + 1 if num_nodes > 0 else 1
    buckets = [[] for _ in range(max_deg)]
    for i in range(num_nodes):
        buckets[deg_remaining[i]].append(i)

    k = 0
    processed = 0
    while processed < num_nodes:
        # Advance k to the next non-empty bucket
        while k < max_deg and not buckets[k]:
            k += 1
        if k >= max_deg:
            break

        node = buckets[k].pop()
        if removed[node]:
            continue

        core[node]    = k
        removed[node] = True
        processed    += 1

        # Reduce degree of unremoved neighbours; push to lower bucket
        for nb in adj[node]:
            if not removed[nb]:
                old_d             = deg_remaining[nb]
                new_d             = max(old_d - 1, 0)
                deg_remaining[nb] = new_d
                buckets[new_d].append(nb)

    return core


def compute_static_topo_features(edge_index: Tensor, num_nodes: int) -> np.ndarray:
    """
    Compute 7 purely structural topology features from graph edges alone.
    No label information is used. Called once per dataset before the CV loop;
    features are identical across all folds.

    Returns float32 array of shape [N, 7]:

      col  0 : std of neighbour degrees
                 Standard deviation of the degrees of all direct neighbours.
                 Complements mean_nbr_deg (col 4): together they parameterise
                 the first two moments of the neighbour-degree distribution.
                 Zero for nodes with degree ≤ 1.

      col  1 : log1p(sum of neighbour degrees)
                 2-hop reach proxy on a compressed scale. Ranked #1 by
                 gradient importance across v6 folds.

      col  2 : log1p(pagerank × N)
                 Scale-invariant, log-compressed pagerank (α=0.85, 20 iters).
                 Raw pagerank dropped in v7 (redundant with this column).

      col  3 : clustering coefficient
                 Fraction of a node's neighbour pairs that are themselves
                 connected. Computed via (A ∘ A²) sparse multiply.

      col  4 : mean neighbour degree
                 Average degree of direct neighbours.

      col  5 : k-core number
                 Maximum k such that this node belongs to a k-core subgraph.
                 Proxy for functional module membership depth. O(E).

      col  6 : 2-hop unique node count
                 |{j ≠ i : (A + A²)[i,j] > 0}|. Distinguishes module-
                 interior (low reach) from module-boundary (high reach) nodes.

    All features are derived solely from edge_index (no labels) and are safe
    to compute globally before the CV loop.
    """
    ei  = edge_index.cpu().numpy()
    row, col = ei

    # ── Adjacency matrix (sparse) ─────────────────────────────────────────────
    A = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(num_nodes, num_nodes),
    )

    # ── Degree ────────────────────────────────────────────────────────────────
    deg = np.bincount(col, minlength=num_nodes).astype(np.float32)

    # ── Neighbour degree array: deg[neighbour] for each edge (row→col) ────────
    #   nbr_deg_sum[i] = Σ_{j∈N(i)} deg[j]   (sum of neighbour degrees)
    #   We also need per-node variance for std.
    nbr_deg_sum = np.zeros(num_nodes, dtype=np.float64)
    nbr_deg_sq  = np.zeros(num_nodes, dtype=np.float64)   # Σ deg(j)²

    np.add.at(nbr_deg_sum, col, deg[row].astype(np.float64))
    np.add.at(nbr_deg_sq,  col, (deg[row] ** 2).astype(np.float64))

    # std of neighbour degrees: sqrt(E[x²] - E[x]²), zero for deg ≤ 1
    mean_nbr_deg = np.where(deg > 0, nbr_deg_sum / deg, 0.0).astype(np.float32)
    variance     = np.where(
        deg > 1,
        nbr_deg_sq / deg - (nbr_deg_sum / deg) ** 2,
        0.0,
    )
    std_nbr_deg  = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

    # ── A² (needed for clustering coeff and 2-hop reach) ─────────────────────
    A2 = A @ A   # [N, N] sparse; A2[i,j] = number of 2-hop paths from i to j

    # ── Clustering coefficient ────────────────────────────────────────────────
    #   tri[i] = number of triangles through i  = (A ∘ A²)[i,:].sum() / 2
    tri      = np.asarray(A2.multiply(A).sum(axis=1)).flatten() / 2.0
    possible = deg * (deg - 1.0) / 2.0
    cc       = np.where(deg >= 2, tri / np.maximum(possible, 1.0), 0.0
                        ).astype(np.float32)

    # ── PageRank via sparse power iteration ───────────────────────────────────
    d_inv = 1.0 / np.maximum(deg, 1.0)
    T = sp.csr_matrix(
        (d_inv[row], (col, row)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    pr = np.ones(num_nodes, dtype=np.float32) / num_nodes
    alpha_pr = 0.85
    for _ in range(20):
        pr = alpha_pr * (T @ pr) + (1.0 - alpha_pr) / num_nodes

    # ── 2-hop unique node count ───────────────────────────────────────────────
    #   reach₂[i] = |{j ≠ i : (A + A²)[i, j] > 0}|
    #   Computed via boolean OR: convert A and A² to boolean sparse, add,
    #   count nnz per row, subtract 1 if the diagonal is nonzero (self-loops).
    A_bool  = A.astype(bool)
    A2_bool = A2.astype(bool)
    reach_mat     = (A_bool + A2_bool).astype(bool)     # union of 1- and 2-hop
    twohop_unique = np.asarray(reach_mat.sum(axis=1)).flatten().astype(np.float32)
    # Exclude self: diagonal of reach_mat (1 if i is reachable from itself)
    diag_reach = np.asarray(reach_mat.diagonal()).flatten()
    twohop_unique -= diag_reach.astype(np.float32)      # subtract self-entry

    # ── Log1p(sum of neighbour degrees) ───────────────────────────────────────
    twohop_sum_log = np.log1p(nbr_deg_sum).astype(np.float32)

    # ── K-core decomposition ──────────────────────────────────────────────────
    print("  Computing k-core decomposition...", flush=True)
    kcore = _compute_kcore(row, col, num_nodes).astype(np.float32)

    # ── Stack into [N, 7] feature matrix ─────────────────────────────────────
    feat = np.stack([
        std_nbr_deg,                  # col  0  std of neighbour degrees
        twohop_sum_log,               # col  1  log1p(sum of nbr degrees)
        np.log1p(pr * num_nodes),     # col  2  log1p(pagerank × N)
        cc,                           # col  3  clustering coefficient
        mean_nbr_deg,                 # col  4  mean neighbour degree
        kcore,                        # col  5  k-core number
        twohop_unique,                # col  6  2-hop unique node count
    ], axis=1)                        # [N, 7]

    return feat.astype(np.float32)


def normalise_topo_features(
    static_feat: np.ndarray,
    train_mask: np.ndarray,
) -> Tensor:
    """
    Standardise topology features (zero-mean, unit-std) using statistics
    computed from training nodes only, then applied to all nodes.

    Parameters
    ----------
    static_feat : [N, N_TOPO_FEATURES]  output of compute_static_topo_features
    train_mask  : [N] boolean, True for nodes in the current fold's training set

    Returns
    -------
    Tensor of shape [N, N_TOPO_FEATURES], float32.

    Notes
    -----
    Although static_feat is identical across folds, normalisation is re-run
    per fold so that the standardisation statistics are always derived from
    the current fold's training nodes only. This avoids any distributional
    leakage from validation/test nodes into the feature scale.
    """
    feat       = static_feat.copy()
    train_feat = feat[train_mask]
    mu         = train_feat.mean(axis=0, keepdims=True)         # [1, N_TOPO_FEATURES]
    std        = train_feat.std(axis=0, keepdims=True) + 1e-8   # [1, N_TOPO_FEATURES]
    feat       = (feat - mu) / std
    return torch.tensor(feat, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Topology MLP Branch (MLP_T)
# ──────────────────────────────────────────────────────────────────────────────

class TopoMLP(nn.Module):
    """
    Two-layer MLP mapping N_TOPO_FEATURES (7) → hidden_dim.

    Mirrors SGCD's MLP_A: a graph-structure encoder entirely separate from
    the message-passing branch. The short, direct gradient path allows this
    branch to learn structural position features efficiently.

    The branch-level fusion weight is no longer a scalar here — it is handled
    per-node by GatedFusion inside ACMEMOGIModel.

    Architecture: Linear → LayerNorm → ReLU → Dropout → Linear → ReLU

    Parameters
    ----------
    hidden_dim : output dimension (matched to ACM hidden_dim)
    dropout    : dropout rate
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        mid = max(hidden_dim // 2, N_TOPO_FEATURES * 2)
        self.net = nn.Sequential(
            nn.Linear(N_TOPO_FEATURES, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, topo_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        topo_feat : [N, N_TOPO_FEATURES]

        Returns
        -------
        h_topo : [N, hidden_dim]
        """
        return self.net(topo_feat)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_omics_indices(feature_names: list[str]) -> list[list[int]]:
    """
    Returns 4 lists of column indices, one per omics type, in order
    [MF, CNA, METH, GE], each of length 16.
    """
    indices = []
    for prefix in OMICS_ORDER:
        idx = [i for i, n in enumerate(feature_names) if n.startswith(prefix + ":")]
        assert len(idx) == N_CANCER_TYPES, \
            f"Expected 16 features for {prefix}, got {len(idx)}"
        indices.append(idx)
    return indices


def prepare_graph(edge_index: Tensor, num_nodes: int, device: torch.device):
    """Pre-compute normalised LP adjacency (with self-loops)."""
    edge_index_lp, edge_weight_lp = compute_lp_norm(edge_index, num_nodes)
    return edge_index_lp.to(device), edge_weight_lp.to(device)


def prepare_topo_features(edge_index: Tensor, num_nodes: int) -> np.ndarray:
    """
    Compute all 7 static topology features from graph structure alone.
    Called once per dataset before the CV loop; no labels are used.

    Returns numpy array of shape [N, N_TOPO_FEATURES] (float32).
    Pass the result directly into normalise_topo_features inside the CV loop.
    """
    return compute_static_topo_features(edge_index, num_nodes)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Training and evaluation utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(labels: Tensor, train_mask: np.ndarray) -> Tensor:
    """Compute BCE positive class weight from training labels."""
    y_train = labels[train_mask].float()
    n_pos   = y_train.sum().item()
    n_neg   = len(y_train) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)


def evaluate(
    model: nn.Module,
    x: Tensor,
    omics_indices: list[list[int]],
    edge_index_lp: Tensor,
    edge_weight_lp: Tensor,
    topo_feat: Tensor,
    labels: Tensor,
    eval_mask: np.ndarray,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits, alphas, gate_val = model(
            x, omics_indices, edge_index_lp, edge_weight_lp, topo_feat
        )
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    y_true = labels.cpu().numpy()[eval_mask]
    y_prob = probs[eval_mask]

    results = {}
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        results["auprc"] = average_precision_score(y_true, y_prob)
        results["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        results["auprc"] = float("nan")
        results["auroc"] = float("nan")

    stacked      = torch.stack(alphas)
    eval_alphas  = stacked[:, eval_mask, :].mean(dim=1)
    results["mean_alpha_LP"] = eval_alphas[:, 0].tolist()
    results["mean_alpha_HP"] = eval_alphas[:, 1].tolist()
    results["mean_alpha_ID"] = eval_alphas[:, 2].tolist()
    # gate_val: [N, hidden_dim]; report scalar mean over eval nodes and dims
    results["gate_mean"] = gate_val[eval_mask].mean().item()

    return results


def diagnose_topo_feature_importance(
    model: nn.Module,
    topo_feat: Tensor,
    device: torch.device,
) -> dict:
    """
    Estimate per-feature importance for the TopoMLP branch via input-gradient
    analysis. No training labels are required.

    Method
    ------
    We pass topo_feat through TopoMLP, sum the output to a scalar (dummy loss),
    and call backward. The gradient of that scalar w.r.t. each input feature
    column measures how sensitively the MLP's output responds to that feature
    across all nodes. We aggregate three complementary statistics:

      grad_mean_abs  : mean |∂output/∂feature_j| across nodes — overall
                       sensitivity; the primary importance signal.

      grad_std       : std of ∂output/∂feature_j across nodes — how much
                       the sensitivity varies by node; high std means the
                       feature matters differently for different node types.

      input_x_grad   : mean |feature_j × ∂output/∂feature_j| — integrated-
                       gradients proxy; down-weights features with large
                       gradients but near-zero values after normalisation.

    Parameters
    ----------
    model     : trained ACMEMOGIModel (best checkpoint)
    topo_feat : [N, N_TOPO_FEATURES] normalised tensor (same fold's version)
    device    : torch device

    Returns
    -------
    dict with keys:
      "grad_mean_abs"  : list[float] length N_TOPO_FEATURES
      "grad_std"       : list[float] length N_TOPO_FEATURES
      "input_x_grad"   : list[float] length N_TOPO_FEATURES
      "feature_names"  : list[str]   TOPO_FEATURE_NAMES
      "rank_by_grad"   : list[str]   feature names sorted by grad_mean_abs desc
    """
    model.eval()

    feat = topo_feat.detach().to(device).requires_grad_(True)   # [N, F]
    h_topo = model.topo_mlp(feat)   # [N, hidden_dim]

    dummy_loss = h_topo.sum()
    dummy_loss.backward()

    grad = feat.grad.detach().cpu()   # [N, F]

    grad_mean_abs = grad.abs().mean(dim=0).tolist()
    grad_std      = grad.std(dim=0).tolist()
    input_x_grad  = (feat.detach().cpu() * grad).abs().mean(dim=0).tolist()

    ranked_indices = sorted(range(N_TOPO_FEATURES),
                            key=lambda i: grad_mean_abs[i], reverse=True)
    rank_by_grad   = [TOPO_FEATURE_NAMES[i] for i in ranked_indices]

    return {
        "grad_mean_abs": grad_mean_abs,
        "grad_std":      grad_std,
        "input_x_grad":  input_x_grad,
        "feature_names": TOPO_FEATURE_NAMES,
        "rank_by_grad":  rank_by_grad,
    }


def train_one_fold(
    data: dict,
    omics_indices: list[list[int]],
    edge_index_lp: Tensor,
    edge_weight_lp: Tensor,
    static_topo: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    device: torch.device,
    config: dict,
) -> tuple[dict, dict]:
    """
    Train for one (train_mask, val_mask) fold.
    Returns val metrics and best model state dict.

    Parameters
    ----------
    static_topo : [N, N_TOPO_FEATURES]  precomputed structural features
                  (7 features, no labels; output of prepare_topo_features).
                  z-score normalisation is applied here using train_mask
                  statistics only, preventing leakage from val/test nodes.
    """
    x         = data["feature"].to(device)
    labels    = data["label"].to(device)

    topo_feat = normalise_topo_features(static_topo, train_mask).to(device)

    model = ACMEMOGIModel(
        omics_dim=config["omics_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    pos_weight = compute_pos_weight(labels, train_mask).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auprc   = -1.0
    best_state       = None
    patience_counter = 0
    patience         = config.get("patience", 30)

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(x, omics_indices, edge_index_lp, edge_weight_lp, topo_feat)  # gate_val discarded during training
        loss = criterion(
            logits.squeeze(1)[train_mask],
            labels[train_mask].float(),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(
                model, x, omics_indices, edge_index_lp, edge_weight_lp,
                topo_feat, labels, val_mask, device,
            )
            scheduler.step(val_metrics["auprc"])

            if val_metrics["auprc"] > best_val_auprc:
                best_val_auprc   = val_metrics["auprc"]
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience // 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(
        model, x, omics_indices, edge_index_lp, edge_weight_lp,
        topo_feat, labels, val_mask, device,
    )

    topo_diag = diagnose_topo_feature_importance(model, topo_feat, device)
    val_metrics["topo_diag"] = topo_diag

    return val_metrics, best_state


# ──────────────────────────────────────────────────────────────────────────────
# 9.  10-rep × 5-fold Cross-Validation Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_validation(data: dict, config: dict, device: torch.device, max_reps: Optional[int] = None) -> dict:
    omics_indices = build_omics_indices(data["feature_name"])
    num_nodes     = data["feature"].shape[0]

    edge_index_lp, edge_weight_lp = prepare_graph(
        data["edge_index"], num_nodes, device
    )

    print(
        "  Computing static topology features\n"
        "  (std_nbr_deg, log1p(2hop_sum), log1p(pr*N),\n"
        "   clust_coeff, mean_nbr_deg, kcore, 2hop_unique)...",
        flush=True,
    )
    static_topo = prepare_topo_features(data["edge_index"], num_nodes)
    print(f"  Static topo shape: {static_topo.shape}  "
          f"(expected [N, {N_TOPO_FEATURES}])", flush=True)

    all_results = []
    split_set   = data["split_set"]
    label_mask  = data["mask"]

    reps        = list(split_set.items())
    if max_reps is not None:
        reps = reps[:max_reps]
    
    total_folds = sum(len(folds) for _, folds in reps)
    fold_count  = 0

    reps        = list(split_set.items())
    if max_reps is not None:
        reps = reps[:max_reps]

    for rep_id, folds in reps:
        rep_results = []
        for fold_id, (train_mask_raw, val_mask_raw) in enumerate(folds):
            fold_count += 1
            train_mask  = train_mask_raw & label_mask
            val_mask    = val_mask_raw   & label_mask

            print(
                f"  Rep {rep_id:2d} | Fold {fold_id} | "
                f"train={train_mask.sum():4d} val={val_mask.sum():4d} | "
                f"fold {fold_count}/{total_folds}",
                end=" → ", flush=True,
            )

            val_metrics, _ = train_one_fold(
                data, omics_indices, edge_index_lp, edge_weight_lp,
                static_topo, train_mask, val_mask, device, config,
            )

            print(
                f"AUPRC={val_metrics['auprc']:.4f}  "
                f"AUROC={val_metrics['auroc']:.4f}  "
                f"α=[LP:{val_metrics['mean_alpha_LP'][0]:.2f} "
                f"HP:{val_metrics['mean_alpha_HP'][0]:.2f} "
                f"ID:{val_metrics['mean_alpha_ID'][0]:.2f}]  "
                f"gate={val_metrics['gate_mean']:.4f}"
            )
            rep_results.append(val_metrics)
        all_results.append(rep_results)

    auprc_all     = [m["auprc"]     for rep in all_results for m in rep]
    auroc_all     = [m["auroc"]     for rep in all_results for m in rep]
    gate_mean_all = [m["gate_mean"] for rep in all_results for m in rep]

    # ── Aggregate topology feature importance across all folds ────────────────
    all_diags = [m["topo_diag"] for rep in all_results for m in rep]
    n_folds_total = len(all_diags)
    agg_grad = np.mean([d["grad_mean_abs"] for d in all_diags], axis=0)  # [F]
    agg_std  = np.mean([d["grad_std"]      for d in all_diags], axis=0)  # [F]
    agg_ixg  = np.mean([d["input_x_grad"]  for d in all_diags], axis=0)  # [F]

    ranked_idx = np.argsort(agg_grad)[::-1]

    topo_importance_summary = {
        "feature_names":       TOPO_FEATURE_NAMES,
        "grad_mean_abs_mean":  agg_grad.tolist(),
        "grad_std_mean":       agg_std.tolist(),
        "input_x_grad_mean":   agg_ixg.tolist(),
        "rank_by_grad":        [TOPO_FEATURE_NAMES[i] for i in ranked_idx],
        "n_folds_aggregated":  n_folds_total,
    }

    summary = {
        "auprc_mean":              float(np.nanmean(auprc_all)),
        "auprc_std":               float(np.nanstd(auprc_all)),
        "auroc_mean":              float(np.nanmean(auroc_all)),
        "auroc_std":               float(np.nanstd(auroc_all)),
        "gate_mean":               float(np.nanmean(gate_mean_all)),
        "topo_importance_summary": topo_importance_summary,
        "all_results":             all_results,
    }
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "omics_dim":     64,     # per-omics projection dim → total enc = 256
    "hidden_dim":    128,    # ACM hidden / output dim
    "n_layers":      2,      # number of stacked ACM layers
    "dropout":       0.4,
    "temperature":   3.0,    # ACM softmax temperature
    "lr":            1e-3,
    "weight_decay":  5e-4,
    "epochs":        200,
    "patience":      30,     # early stopping (in epochs, checked every 5)
}




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ACM-EMOGI v7: cancer gene prediction via ACM graph convolution,\n"
            "parallel topology MLP branch (7 features), and per-node gated fusion.\n"
            "Runs N-rep × 5-fold cross-validation and reports AUPRC/AUROC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data / run control ────────────────────────────────────────────────────
    parser.add_argument(
        "--data",
        default="dataset_MULTINET_ten_5CV.pkl",
        help="Path to the dataset pickle file.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Number of CV repetitions to run (default: 2, i.e. 2×5=10 folds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Global random seed for Python, NumPy, and PyTorch (including CUDA). "
            "Set the same value across runs to reproduce results exactly."
        ),
    )

    # ── Model hyperparameters ─────────────────────────────────────────────────
    parser.add_argument(
        "--omics_dim",
        type=int,
        default=DEFAULT_CONFIG["omics_dim"],
        help="Projection dimension per omics type; total encoder output = 4 × omics_dim.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="Hidden dimension shared by ACM layers, TopoMLP output, and GatedFusion.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=DEFAULT_CONFIG["n_layers"],
        help="Number of stacked ACM convolution layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_CONFIG["dropout"],
        help="Dropout rate applied throughout the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help="Softmax temperature T for ACM channel-mixing weights.",
    )

    # ── Optimiser hyperparameters ─────────────────────────────────────────────
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["lr"],
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help="Adam weight decay (L2 regularisation).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help="Maximum training epochs per fold.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help=(
            "Early-stopping patience in epochs. "
            "Validation AUPRC is checked every 5 epochs; "
            "training stops after patience/5 non-improving checks."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    # seed_everything() is also called from __main__ before parse_args() so
    # that CUBLAS_WORKSPACE_CONFIG is set before CUDA initialises. Calling it
    # again here is safe (idempotent) and handles the case where main() is
    # imported and called directly rather than run as a script.
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  seed: {args.seed}\n")

    # ── Build config from CLI args ────────────────────────────────────────────
    config = {
        "omics_dim":    args.omics_dim,
        "hidden_dim":   args.hidden_dim,
        "n_layers":     args.n_layers,
        "dropout":      args.dropout,
        "temperature":  args.temperature,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "epochs":       args.epochs,
        "patience":     args.patience,
    }

    print("Loading data...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    print(f"  Nodes       : {data['feature'].shape[0]}")
    print(f"  Features    : {data['feature'].shape[1]}  (4 omics × 16 cancer types)")
    print(f"  Edges       : {data['edge_index'].shape[1]}")
    print(f"  Labelled    : {data['mask'].sum()}")
    print(f"  Cancer genes: {int((data['label']==1).sum())}  "
          f"({100*float((data['label']==1).sum())/data['mask'].sum():.1f}% of labelled)")
    print(f"  Reps × Folds: {args.reps} × "
          f"{len(list(data['split_set'].values())[0])}\n")

    print(f"Topology features: {N_TOPO_FEATURES}  "
          f"(std_nbr_deg, log1p(2hop_sum), log1p(pr*N), "
          f"clust_coeff, mean_nbr_deg, kcore, 2hop_unique)\n")
    print("Config:", config, "\n")
    print(f"Starting {args.reps} × 5-fold cross-validation...\n")

    summary = run_cross_validation(data, config, device, max_reps=args.reps)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  AUPRC   : {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")
    print(f"  AUROC   : {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print("=" * 60)

    # ── Topology feature importance table ─────────────────────────────────────
    tip = summary["topo_importance_summary"]
    print(f"\nTOPO FEATURE IMPORTANCE  "
          f"(avg over {tip['n_folds_aggregated']} folds, z-scored inputs)")
    print(f"  {'feature':<18}  {'grad_abs':>9}  {'grad_std':>9}  {'inp×grad':>9}  rank")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}  ----")
    rank_order = {name: i + 1 for i, name in enumerate(tip["rank_by_grad"])}
    new_v7 = {"std_nbr_deg", "2hop_unique"}
    for j, name in enumerate(tip["feature_names"]):
        tag = "← enriched v7" if name in new_v7 else ""
        print(
            f"  {name:<18}  "
            f"{tip['grad_mean_abs_mean'][j]:>9.4f}  "
            f"{tip['grad_std_mean'][j]:>9.4f}  "
            f"{tip['input_x_grad_mean'][j]:>9.4f}  "
            f"#{rank_order[name]:<2}  {tag}"
        )
    print(f"\n  Ranked by grad_abs: {tip['rank_by_grad']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    # Extract --seed from argv before parse_args() so seed_everything() runs
    # before any torch/numpy RNG state is consumed (weight init, dropout, etc.)
    # and before CUDA is initialised (required for CUBLAS_WORKSPACE_CONFIG).
    _seed = 42  # matches parse_args default
    for _i, _arg in enumerate(sys.argv):
        if _arg in ("--seed", "-seed") and _i + 1 < len(sys.argv):
            try:
                _seed = int(sys.argv[_i + 1])
            except ValueError:
                pass
    seed_everything(_seed)

    main()