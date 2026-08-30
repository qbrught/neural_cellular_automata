"""Cell-level functional class divergence.

Each cell is a response map (its own ψ / f). This module evaluates every
cell on a frozen probe bank, then reports pairwise distances, Δ, k-means
ARI vs goal labels, and 2D embeddings.

See research/FUNCTIONAL_DIVERGENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from parameters import Parameters, batched_mlp, f_in_dim, psi_in_dim, psi_out_dim
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, State


# ---------------------------------------------------------------------------
# Frozen probe banks (v1). Changing these is a new metric version.
# ---------------------------------------------------------------------------

VOTE_PROBES: tuple[dict[str, Any], ...] = (
    {"name": "kin_alive_blank", "relation": "kin", "x_r": 1.0, "s_scale": 0.0},
    {"name": "foe_alive_blank", "relation": "foe", "x_r": 1.0, "s_scale": 0.0},
    {"name": "kin_dead_blank", "relation": "kin", "x_r": 0.0, "s_scale": 0.0},
    {"name": "foe_dead_blank", "relation": "foe", "x_r": 0.0, "s_scale": 0.0},
    {"name": "kin_alive_exc", "relation": "kin", "x_r": 1.0, "s_scale": 0.5},
    {"name": "foe_alive_exc", "relation": "foe", "x_r": 1.0, "s_scale": 0.5},
)

WEIGHT_PROBES: tuple[dict[str, Any], ...] = (
    {"name": "alive_blank", "x_s": 1.0, "x_r": 1.0, "s_r_scale": 0.0},
    {"name": "alive_exc", "x_s": 1.0, "x_r": 1.0, "s_r_scale": 0.5},
    {"name": "recv_dead", "x_s": 1.0, "x_r": 0.0, "s_r_scale": 0.0},
)

F_PROBES: tuple[dict[str, Any], ...] = (
    {"name": "alive_M0", "x": 1.0, "M_scale": 0.0},
    {"name": "alive_Mhalf", "x": 1.0, "M_scale": 0.5},
    {"name": "dead_M0", "x": 0.0, "M_scale": 0.0},
)

FAMILY_REALIZED = "realized"
FAMILY_COMMON = "common"
FAMILY_BOTH_HEADS = "both_heads"
FAMILY_WEIGHTS_ONLY = "weights_only"
FAMILY_FULL = "full"
# Agent output (own-goal), shared-question maps, goal-ablated maps.
HEADLINE_FAMILIES = (FAMILY_REALIZED, FAMILY_COMMON, FAMILY_WEIGHTS_ONLY)
COMMON_SENDER_GOALS: tuple[tuple[str, float], ...] = (
    ("R", 0.0),  # GOAL_REPRODUCE
    ("E", 1.0),  # GOAL_ELIMINATE
)


def vote_probe_names() -> list[str]:
    return [p["name"] for p in VOTE_PROBES]


def kin_probe_mask() -> np.ndarray:
    return np.array([p["relation"] == "kin" for p in VOTE_PROBES], dtype=bool)


def common_probe_names() -> list[str]:
    return [f"{lab}:{p['name']}" for lab, _ in COMMON_SENDER_GOALS for p in VOTE_PROBES]


def common_kin_mask() -> np.ndarray:
    return np.tile(kin_probe_mask(), len(COMMON_SENDER_GOALS))


# ---------------------------------------------------------------------------
# Probe tensors
# ---------------------------------------------------------------------------

def _psi_layout_fill(
    psi_in: Tensor,
    k: int,
    s_s: Tensor,
    h_s: Tensor,
    x_s: Tensor,
    g_s: Tensor,
    s_r: Tensor,
    h_r: Tensor,
    x_r: Tensor,
    g_r: Tensor,
) -> None:
    """Write probe k into psi_in[..., k, :] with layout [s_s,h_s,x_s,g_s,s_r,h_r,x_r,g_r]."""
    d = s_s.shape[-1]
    off = 0
    psi_in[:, :, k, off : off + d] = s_s
    off += d
    psi_in[:, :, k, off : off + d] = h_s
    off += d
    psi_in[:, :, k, off] = x_s
    off += 1
    psi_in[:, :, k, off] = g_s
    off += 1
    psi_in[:, :, k, off : off + d] = s_r
    off += d
    psi_in[:, :, k, off : off + d] = h_r
    off += d
    psi_in[:, :, k, off] = x_r
    off += 1
    psi_in[:, :, k, off] = g_r


def build_vote_probe_inputs(goals: Tensor, d: int) -> Tensor:
    """ψ inputs for the frozen vote-probe bank.

    Sender: s=0, h=0, x=1, g=own. Receiver per VOTE_PROBES.
    Shape (N, N, K, 4d+4).
    """
    N = int(goals.shape[0])
    device = goals.device
    dtype = torch.float32
    g_s = goals.to(device=device, dtype=dtype)
    zeros_d = torch.zeros(N, N, d, device=device, dtype=dtype)
    ones = torch.ones(N, N, device=device, dtype=dtype)
    K = len(VOTE_PROBES)
    psi_in = torch.zeros(N, N, K, psi_in_dim(d), device=device, dtype=dtype)
    for k, spec in enumerate(VOTE_PROBES):
        g_r = g_s if spec["relation"] == "kin" else (1.0 - g_s)
        s_r = torch.full((N, N, d), float(spec["s_scale"]), device=device, dtype=dtype)
        x_r = torch.full((N, N), float(spec["x_r"]), device=device, dtype=dtype)
        _psi_layout_fill(
            psi_in, k,
            zeros_d, zeros_d, ones, g_s,
            s_r, zeros_d, x_r, g_r,
        )
    return psi_in


def build_common_probe_inputs(N: int, d: int, device: torch.device | str) -> Tensor:
    """ψ inputs: six vote probes × two canonical sender goals.

    Every cell sees the same 12 situations (g_s is not the cell's own goal).
    Order: all VOTE_PROBES with g_s=0 (R), then all with g_s=1 (E).
    Shape (N, N, 12, 4d+4).
    """
    dtype = torch.float32
    zeros_d = torch.zeros(N, N, d, device=device, dtype=dtype)
    ones = torch.ones(N, N, device=device, dtype=dtype)
    K = len(VOTE_PROBES) * len(COMMON_SENDER_GOALS)
    psi_in = torch.zeros(N, N, K, psi_in_dim(d), device=device, dtype=dtype)
    slot = 0
    for _, g_s_val in COMMON_SENDER_GOALS:
        g_s = torch.full((N, N), float(g_s_val), device=device, dtype=dtype)
        for spec in VOTE_PROBES:
            g_r = g_s if spec["relation"] == "kin" else (1.0 - g_s)
            s_r = torch.full(
                (N, N, d), float(spec["s_scale"]), device=device, dtype=dtype
            )
            x_r = torch.full((N, N), float(spec["x_r"]), device=device, dtype=dtype)
            _psi_layout_fill(
                psi_in, slot,
                zeros_d, zeros_d, ones, g_s,
                s_r, zeros_d, x_r, g_r,
            )
            slot += 1
    return psi_in


def build_weight_probe_inputs(N: int, d: int, device: torch.device | str) -> Tensor:
    """ψ inputs with all goal bits zeroed (weight-only fingerprint).

    Shape (N, N, K, 4d+4).
    """
    dtype = torch.float32
    zeros_d = torch.zeros(N, N, d, device=device, dtype=dtype)
    zeros = torch.zeros(N, N, device=device, dtype=dtype)
    K = len(WEIGHT_PROBES)
    psi_in = torch.zeros(N, N, K, psi_in_dim(d), device=device, dtype=dtype)
    for k, spec in enumerate(WEIGHT_PROBES):
        x_s = torch.full((N, N), float(spec["x_s"]), device=device, dtype=dtype)
        x_r = torch.full((N, N), float(spec["x_r"]), device=device, dtype=dtype)
        s_r = torch.full(
            (N, N, d), float(spec["s_r_scale"]), device=device, dtype=dtype
        )
        _psi_layout_fill(
            psi_in, k,
            zeros_d, zeros_d, x_s, zeros,
            s_r, zeros_d, x_r, zeros,
        )
    return psi_in


def build_f_probe_inputs(
    goals: Tensor,
    d: int,
    *,
    goal_in_f: bool,
) -> Tensor:
    """f inputs for F_PROBES. Shape (N, N, K, 3d+2) = [s, h, x, M, goal_slot]."""
    N = int(goals.shape[0])
    device = goals.device
    dtype = torch.float32
    zeros_d = torch.zeros(N, N, d, device=device, dtype=dtype)
    if goal_in_f:
        goal_slot = goals.to(device=device, dtype=dtype)
    else:
        goal_slot = torch.zeros(N, N, device=device, dtype=dtype)
    K = len(F_PROBES)
    f_in = torch.zeros(N, N, K, f_in_dim(d), device=device, dtype=dtype)
    for k, spec in enumerate(F_PROBES):
        x = torch.full((N, N), float(spec["x"]), device=device, dtype=dtype)
        M = torch.full((N, N, d), float(spec["M_scale"]), device=device, dtype=dtype)
        off = 0
        f_in[:, :, k, off : off + d] = zeros_d
        off += d
        f_in[:, :, k, off : off + d] = zeros_d
        off += d
        f_in[:, :, k, off] = x
        off += 1
        f_in[:, :, k, off : off + d] = M
        off += d
        f_in[:, :, k, off] = goal_slot
    return f_in


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def psi_on_probes(params: Parameters, psi_in: Tensor) -> Tensor:
    """Apply each cell's ψ to K probes. psi_in (N,N,K,in) -> (N,N,K,d+2)."""
    return batched_mlp(
        psi_in, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2
    )


def f_on_probes(params: Parameters, f_in: Tensor) -> Tensor:
    """Apply each cell's f to K probes. f_in (N,N,K,in) -> (N,N,K,2d)."""
    return batched_mlp(
        f_in, params.f_W1, params.f_b1, params.f_W2, params.f_b2
    )


def realized_votes(
    psi_out: Tensor,
    typed_votes: bool,
    kin: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Vote that reaches the receiver under this version's routing.

    psi_out: (N, N, K, d+2). Default K matches VOTE_PROBES; pass `kin` when K differs
    (e.g. common-domain 12-probe bank).
    Returns (N, N, K).
    """
    d = psi_out.shape[-1] - 2
    help_v = psi_out[..., d]
    harm_v = psi_out[..., d + 1]
    if not typed_votes:
        return help_v
    K = int(psi_out.shape[2])
    if kin is None:
        if K != len(VOTE_PROBES):
            raise ValueError(
                f"kin mask required when K={K} != len(VOTE_PROBES)={len(VOTE_PROBES)}"
            )
        kin_t = torch.tensor(
            [p["relation"] == "kin" for p in VOTE_PROBES],
            device=psi_out.device,
            dtype=torch.bool,
        )
    else:
        kin_t = torch.as_tensor(kin, device=psi_out.device, dtype=torch.bool).reshape(-1)
        if int(kin_t.numel()) != K:
            raise ValueError(f"kin mask length {int(kin_t.numel())} != K={K}")
    return torch.where(kin_t.view(1, 1, -1), help_v, harm_v)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """(n, p) -> (n, n) Euclidean distances."""
    X = np.asarray(X, dtype=float)
    sq = np.einsum("ij,ij->i", X, X)
    gram = X @ X.T
    dist2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2, dtype=float)


def _pair_mean(D: np.ndarray, a: np.ndarray, b: np.ndarray, *, same: bool) -> float:
    ia = np.flatnonzero(a)
    ib = np.flatnonzero(b)
    if same:
        n = ia.size
        if n < 2:
            return float("nan")
        block = D[np.ix_(ia, ia)]
        iu = np.triu_indices(n, k=1)
        return float(block[iu].mean())
    if ia.size == 0 or ib.size == 0:
        return float("nan")
    return float(D[np.ix_(ia, ib)].mean())


def _nan_class_stats() -> dict[str, float]:
    return {
        "n_r": 0.0,
        "n_e": 0.0,
        "within_r": float("nan"),
        "within_e": float("nan"),
        "within": float("nan"),
        "between": float("nan"),
        "delta": float("nan"),
        "n_classes_present": 0.0,
    }


def class_distance_stats(
    D: np.ndarray,
    goals: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Between/within class mean distances on a pairwise matrix D.

    ``within`` is the equal average of the two type-conditional pairwise
    means, so ``delta = between - within`` is the sample energy distance
    ``E/2`` (Székely–Rizzo). ``delta`` is nan unless both types have at
    least two cells (otherwise a type-conditional law has no pair).
    """
    g = np.asarray(goals).reshape(-1)
    n = g.size
    if n == 0:
        return _nan_class_stats()
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
    is_r = mask & (g == GOAL_REPRODUCE)
    is_e = mask & (g == GOAL_ELIMINATE)
    within_r = _pair_mean(D, is_r, is_r, same=True)
    within_e = _pair_mean(D, is_e, is_e, same=True)
    between = _pair_mean(D, is_r, is_e, same=False)
    n_r = int(is_r.sum())
    n_e = int(is_e.sum())
    if n_r >= 2 and n_e >= 2 and np.isfinite(within_r) and np.isfinite(within_e):
        within = 0.5 * (within_r + within_e)
    elif n_r >= 2:
        within = within_r
    elif n_e >= 2:
        within = within_e
    else:
        within = float("nan")
    # Energy contrast needs both type-conditional self-distances.
    delta = (
        float(between - within)
        if (n_r >= 2 and n_e >= 2 and np.isfinite(between) and np.isfinite(within))
        else float("nan")
    )
    return {
        "n_r": float(n_r),
        "n_e": float(n_e),
        "within_r": float(within_r),
        "within_e": float(within_e),
        "within": float(within),
        "between": float(between),
        "delta": delta,
        "n_classes_present": float(int(n_r > 0) + int(n_e > 0)),
    }


def kmeans_ari(X: np.ndarray, labels: np.ndarray, *, random_state: int = 0) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    uniq = np.unique(labels)
    out = {
        "ari_kmeans_k2": float("nan"),
        "silhouette_goal": float("nan"),
        "silhouette_kmeans": float("nan"),
        "kmeans_cluster": None,
    }
    if n < 4 or uniq.size < 2:
        return out
    if not np.isfinite(X).all() or np.allclose(X, X[0:1]):
        return out
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    pred = km.fit_predict(X)
    out["kmeans_cluster"] = pred
    out["ari_kmeans_k2"] = float(adjusted_rand_score(labels, pred))
    # Silhouette needs each label to appear at least once and n > n_labels.
    if np.unique(labels).size >= 2 and n > 2:
        try:
            out["silhouette_goal"] = float(silhouette_score(X, labels, metric="euclidean"))
        except ValueError:
            pass
    if np.unique(pred).size >= 2 and n > 2:
        try:
            out["silhouette_kmeans"] = float(silhouette_score(X, pred, metric="euclidean"))
        except ValueError:
            pass
    return out


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(X)
    return xy, np.asarray(pca.explained_variance_ratio_, dtype=float)


def umap_2d(X: np.ndarray, *, random_state: int = 0) -> np.ndarray | None:
    n = X.shape[0]
    if n < 5:
        return None
    n_neighbors = int(min(15, n - 1))
    try:
        from umap.umap_ import UMAP
    except ImportError:
        try:
            from umap import UMAP  # type: ignore
        except ImportError:
            return None
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(X), dtype=float)


# ---------------------------------------------------------------------------
# Snapshot evaluation
# ---------------------------------------------------------------------------

@dataclass
class FamilyResult:
    name: str
    X: np.ndarray                 # (n, p) raw
    Xz: np.ndarray                # (n, p) z-scored
    feature_names: list[str]
    D: np.ndarray                 # (n, n)
    stats_all: dict[str, float]
    stats_alive: dict[str, float]
    cluster: np.ndarray | None    # (n,) k-means on all cells
    ari_all: float
    sil_goal_all: float
    ari_alive: float
    sil_goal_alive: float
    pca_xy: np.ndarray
    pca_var: np.ndarray
    umap_xy: np.ndarray | None


@dataclass
class SnapshotFunctional:
    """Everything computed for one (params, state) snapshot."""

    n: int
    d: int
    typed_votes: bool
    goals: np.ndarray             # (n,)
    alive: np.ndarray             # (n,) bool
    families: dict[str, FamilyResult] = field(default_factory=dict)
    kin_foe_gap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # realized vote on kin_alive_blank / foe_alive_blank, shape (n,)
    realized_kin_blank: np.ndarray = field(default_factory=lambda: np.zeros(0))
    realized_foe_blank: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def scalars(self) -> dict[str, float]:
        out: dict[str, float] = {
            "n_cells": float(self.n),
            "n_alive": float(self.alive.sum()),
            "n_repro": float((self.goals == GOAL_REPRODUCE).sum()),
            "n_elim": float((self.goals == GOAL_ELIMINATE).sum()),
            "n_repro_alive": float(
                ((self.goals == GOAL_REPRODUCE) & self.alive).sum()
            ),
            "n_elim_alive": float(
                ((self.goals == GOAL_ELIMINATE) & self.alive).sum()
            ),
        }
        gap = self.kin_foe_gap
        if gap.size:
            r = self.goals == GOAL_REPRODUCE
            e = self.goals == GOAL_ELIMINATE
            out["mean_kin_foe_gap_R"] = float(gap[r].mean()) if r.any() else float("nan")
            out["mean_kin_foe_gap_E"] = float(gap[e].mean()) if e.any() else float("nan")
            ra = r & self.alive
            ea = e & self.alive
            out["mean_kin_foe_gap_R_alive"] = (
                float(gap[ra].mean()) if ra.any() else float("nan")
            )
            out["mean_kin_foe_gap_E_alive"] = (
                float(gap[ea].mean()) if ea.any() else float("nan")
            )
        for name, fam in self.families.items():
            out[f"delta_{name}_all"] = fam.stats_all["delta"]
            out[f"delta_{name}_alive"] = fam.stats_alive["delta"]
            out[f"within_{name}_all"] = fam.stats_all["within"]
            out[f"between_{name}_all"] = fam.stats_all["between"]
            out[f"ari_{name}_all"] = fam.ari_all
            out[f"sil_goal_{name}_all"] = fam.sil_goal_all
            out[f"ari_{name}_alive"] = fam.ari_alive
            out[f"sil_goal_{name}_alive"] = fam.sil_goal_alive
        return out


def _flatten(t: Tensor) -> np.ndarray:
    arr = t.detach().cpu().reshape(t.shape[0] * t.shape[1], *t.shape[2:]).numpy()
    return np.asarray(arr, dtype=float)


def _family_from_X(
    name: str,
    X: np.ndarray,
    feature_names: list[str],
    goals: np.ndarray,
    alive: np.ndarray,
    *,
    embed: bool,
    umap: bool = False,
) -> FamilyResult:
    Xz = zscore(X)
    D = pairwise_euclidean(Xz)
    stats_all = class_distance_stats(D, goals, mask=None)
    alive = np.asarray(alive, dtype=bool).reshape(-1)
    n_alive = int(alive.sum())
    # Alive-only geometry is recomputed on the living cells (re-zscore, new D),
    # not a subset of the all-cell distance matrix.
    if n_alive >= 2:
        Xz_alive = zscore(X[alive])
        D_alive = pairwise_euclidean(Xz_alive)
        stats_alive = class_distance_stats(D_alive, goals[alive])
    else:
        Xz_alive = np.zeros((0, X.shape[1]))
        stats_alive = _nan_class_stats()
    km_all = kmeans_ari(Xz, goals)
    if n_alive >= 4 and np.unique(goals[alive]).size >= 2:
        km_alive = kmeans_ari(Xz_alive, goals[alive])
        ari_alive = km_alive["ari_kmeans_k2"]
        sil_alive = km_alive["silhouette_goal"]
    else:
        ari_alive = float("nan")
        sil_alive = float("nan")
    if embed:
        pca_xy, pca_var = pca_2d(Xz)
        umap_xy = umap_2d(Xz) if umap else None
    else:
        pca_xy = np.zeros((X.shape[0], 2))
        pca_var = np.array([np.nan, np.nan])
        umap_xy = None
    cluster = km_all["kmeans_cluster"]
    if cluster is None:
        cluster = np.full(X.shape[0], -1, dtype=int)
    return FamilyResult(
        name=name,
        X=X,
        Xz=Xz,
        feature_names=feature_names,
        D=D,
        stats_all=stats_all,
        stats_alive=stats_alive,
        cluster=np.asarray(cluster, dtype=int),
        ari_all=float(km_all["ari_kmeans_k2"]),
        sil_goal_all=float(km_all["silhouette_goal"]),
        ari_alive=float(ari_alive),
        sil_goal_alive=float(sil_alive),
        pca_xy=pca_xy,
        pca_var=pca_var,
        umap_xy=umap_xy,
    )


@torch.no_grad()
def evaluate_snapshot(
    params: Parameters,
    state: State,
    u: Tensor,
    *,
    typed_votes: bool,
    goal_in_f: bool = False,
    embed: bool = True,
) -> SnapshotFunctional:
    """Build response vectors and geometry for one snapshot."""
    N = state.N
    d = state.d
    device = state.x.device
    goals_t = state.goals
    psi_vote = psi_on_probes(params, build_vote_probe_inputs(goals_t, d))
    realized = realized_votes(psi_vote, typed_votes)
    help_v = psi_vote[..., d]
    harm_v = psi_vote[..., d + 1]
    messages = psi_vote[..., :d]

    psi_common = psi_on_probes(params, build_common_probe_inputs(N, d, device))
    common = realized_votes(
        psi_common, typed_votes, kin=common_kin_mask()
    )

    psi_w = psi_on_probes(params, build_weight_probe_inputs(N, d, device))
    w_help = psi_w[..., d]
    w_harm = psi_w[..., d + 1]

    f_out = f_on_probes(params, build_f_probe_inputs(goals_t, d, goal_in_f=goal_in_f))
    s_prop = f_out[..., :d]
    u_b = u.to(device=device, dtype=s_prop.dtype).view(1, 1, 1, d)
    f_signal = (s_prop * u_b).sum(dim=-1)

    realized_np = _flatten(realized)          # (n, K)
    common_np = _flatten(common)              # (n, 2K)
    help_np = _flatten(help_v)
    harm_np = _flatten(harm_v)
    msg_np = _flatten(messages).reshape(N * N, -1)
    both = np.concatenate([help_np, harm_np], axis=1)
    w_both = np.concatenate([_flatten(w_help), _flatten(w_harm)], axis=1)
    fsig = _flatten(f_signal)
    full = np.concatenate([both, msg_np, fsig], axis=1)

    vote_names = vote_probe_names()
    common_names = common_probe_names()
    both_names = [f"help:{n}" for n in vote_names] + [f"harm:{n}" for n in vote_names]
    w_names = [f"help:{p['name']}" for p in WEIGHT_PROBES] + [
        f"harm:{p['name']}" for p in WEIGHT_PROBES
    ]
    full_names = (
        both_names
        + [f"msg{c}:{n}" for n in vote_names for c in range(d)]
        + [f"f_signal:{p['name']}" for p in F_PROBES]
    )

    goals = goals_t.detach().cpu().reshape(-1).numpy().astype(int)
    alive = (state.x.detach().cpu().reshape(-1).numpy() > 0)

    snap = SnapshotFunctional(
        n=N * N,
        d=d,
        typed_votes=bool(typed_votes),
        goals=goals,
        alive=alive,
        kin_foe_gap=realized_np[:, 0] - realized_np[:, 1],
        realized_kin_blank=realized_np[:, 0],
        realized_foe_blank=realized_np[:, 1],
    )
    specs = (
        (FAMILY_REALIZED, realized_np, vote_names),
        (FAMILY_COMMON, common_np, common_names),
        (FAMILY_BOTH_HEADS, both, both_names),
        (FAMILY_WEIGHTS_ONLY, w_both, w_names),
        (FAMILY_FULL, full, full_names),
    )
    for name, X, fnames in specs:
        snap.families[name] = _family_from_X(
            name,
            X,
            fnames,
            goals,
            alive,
            embed=embed,
            umap=bool(embed and name in HEADLINE_FAMILIES),
        )
    return snap


def compare_init_late(
    late: SnapshotFunctional,
    init: SnapshotFunctional,
) -> dict[str, float]:
    """Δ_late − Δ_init and ARI_late − ARI_init per family."""
    out: dict[str, float] = {}
    for name in late.families:
        if name not in init.families:
            continue
        d_l = late.families[name].stats_all["delta"]
        d_i = init.families[name].stats_all["delta"]
        a_l = late.families[name].ari_all
        a_i = init.families[name].ari_all
        out[f"delta_{name}_all_late"] = d_l
        out[f"delta_{name}_all_init"] = d_i
        out[f"delta_{name}_all_learned"] = (
            float(d_l - d_i) if np.isfinite(d_l) and np.isfinite(d_i) else float("nan")
        )
        out[f"ari_{name}_all_late"] = a_l
        out[f"ari_{name}_all_init"] = a_i
        out[f"ari_{name}_all_learned"] = (
            float(a_l - a_i) if np.isfinite(a_l) and np.isfinite(a_i) else float("nan")
        )
        out[f"delta_{name}_alive_late"] = late.families[name].stats_alive["delta"]
        out[f"ari_{name}_alive_late"] = late.families[name].ari_alive
    return out
