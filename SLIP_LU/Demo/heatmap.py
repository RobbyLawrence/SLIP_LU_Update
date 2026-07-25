"""
Heatmap of entry bit-lengths across a sDIPPS diagonal-push chain.

Left: grid of small heatmaps, one per iteration snapshot. Each cell colored
by ceil(log2(|entry|+1)).
Right: max bit length vs step, with Hadamard bound line
       ceil(m * log2(sigma * sqrt(m))) as reference.

Only diagonal-push chains for now (sparse A that avoids sU != 0).
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUTDIR = "heatmaps"
os.makedirs(OUTDIR, exist_ok=True)

# Suppress the module-level noise from sdipps at import: run its top-level tests
# but discard stdout during import.
import io, contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import sdipps as sd

def bit_len_matrix(M):
    out = np.zeros(M.shape, dtype=int)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v is None:
                continue
            v = int(v)
            if v == 0:
                out[i, j] = 0
            else:
                out[i, j] = int(abs(v)).bit_length()
    return out

def hadamard_bound_bits(m, sigma):
    return math.ceil(m * math.log2(sigma * math.sqrt(m)))

def dense_LL_LU_bigint(P, A, Q):
    """Bigint-safe reimplementation of sdipps.dense_LL_LU."""
    n = A.shape[0]
    LU = np.zeros((n, n), dtype=object)
    for i in range(n):
        LU[i, i] = 1
    for k in range(n):
        col = np.array([int(A[P[i], Q[k]]) for i in range(n)], dtype=object)
        x = col
        for j in range(1, n):
            x[j] = LU[0, 0] * x[j] - LU[j, 0] * x[0]
            for i in range(1, min(j, k)):
                x[j] = (LU[i, i] * x[j] - LU[j, i] * x[i]) // LU[i-1, i-1]
        LU[:, k] = x
    return LU

def run_and_plot(A, K, sigma, title, outpath):
    n = A.shape[0]
    P = list(range(n)); Q = list(range(n))
    with contextlib.redirect_stdout(io.StringIO()):
        LU = dense_LL_LU_bigint(P, A, Q)
    F0 = LU.tolist()
    L, rhos, U = sd.frame_to_LU(F0)
    snaps = [("init LU frame", np.array(LU, dtype=object))]
    orig = sd.dense_LL_LU
    sd.dense_LL_LU = dense_LL_LU_bigint
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sd.SDIPPS(A, K, P, Q, L, U, rhos, snapshots=snaps)
    finally:
        sd.dense_LL_LU = orig
    # Drop the SDIPPS-internal "init" (empty Ftmp) since we replaced it above.
    snaps = [snaps[0]] + [s for s in snaps[1:] if s[0] != "init"]

    bitgrids = [(lbl, bit_len_matrix(F)) for lbl, F in snaps]
    global_max = max(g.max() for _, g in bitgrids) if bitgrids else 1
    bound = hadamard_bound_bits(n, sigma)
    vmax = max(global_max, bound)

    ncols = min(len(bitgrids), 4)
    nrows_grid = int(math.ceil(len(bitgrids) / ncols))
    fig = plt.figure(figsize=(4*ncols + 4, 3*nrows_grid + 0.5))
    gs = fig.add_gridspec(nrows_grid, ncols + 1,
                          width_ratios=[1]*ncols + [1.2])

    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.viridis

    for idx, (lbl, grid) in enumerate(bitgrids):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_title(lbl, fontsize=10)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.tick_params(labelsize=7)
        if n <= 10:
            for i in range(n):
                for j in range(n):
                    if grid[i, j] > 0:
                        ax.text(j, i, str(int(grid[i, j])),
                                ha="center", va="center",
                                color="white" if grid[i, j] > vmax*0.5 else "black",
                                fontsize=7)

    ax_traj = fig.add_subplot(gs[:, ncols])
    step_max = [g.max() for _, g in bitgrids]
    labels = [lbl for lbl, _ in bitgrids]
    xs = list(range(len(step_max)))
    ax_traj.plot(xs, step_max, marker="o", label="observed max bits")
    ax_traj.axhline(bound, color="red", linestyle="--",
                    label=f"Hadamard bound = {bound}")
    ax_traj.set_xticks(xs)
    ax_traj.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_traj.set_ylabel("max bit length")
    ax_traj.set_ylim(0, vmax * 1.1 + 1)
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.3)

    cbar_ax = fig.add_axes([0.05, 0.02, 0.55, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                 label="ceil(log2(|entry|+1))")

    fig.suptitle(f"{title}  (n={n}, K={K}, sigma={sigma})")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    full = os.path.join(OUTDIR, outpath)
    fig.savefig(full, dpi=130)
    plt.close(fig)
    print(f"wrote {full}: {len(bitgrids)} snapshots, obs_max={max(step_max)}, bound={bound}")

# ---- diag-only test cases (same skeletons as sdipps.py D1-D3, larger sigma) ----

A_D1 = np.array([[3, 0, 0, 0, 0],
                 [1, 4, 0, 0, 0],
                 [2, 0, 5, 0, 0],
                 [0, 0, 0, 6, 0],
                 [1, 0, 0, 0, 7]])

A_D2 = np.array([[2, 0, 0, 0, 0, 0],
                 [3, 5, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0],
                 [1, 0, 0, 6, 0, 0],
                 [0, 0, 0, 0, 3, 0],
                 [2, 0, 0, 0, 0, 7]])

A_D3 = np.array([[3, 0, 4, 0, 5],
                 [0, 6, 0, 0, 0],
                 [0, 1, 7, 0, 0],
                 [0, 2, 0, 8, 0],
                 [0, 0, 0, 0, 9]])

# A larger sparse case forcing a longer diag chain
rng = np.random.default_rng(7)
def sparse_diag_chain(n, sigma, seed):
    """Only col 0 and diagonal populated - guarantees pure diag push chain."""
    r = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, i] = r.integers(1, sigma+1)
    for i in range(1, n):
        A[i, 0] = r.integers(-sigma, sigma+1)
    return A

A_big = sparse_diag_chain(8, 20, seed=11)

def dense_lower_diag_chain(n, sigma, seed):
    """Dense lower-triangular integer matrix. U ends up diagonal, so
    muU[k+1] stays 0 forever and the whole chain is diagonal pushes.
    Growth concentrates in L / Ftmp lower half."""
    r = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, i] = r.integers(1, sigma+1)  # nonzero pivot
        for j in range(i):
            A[i, j] = r.integers(-sigma, sigma+1)
    return A

A_dense6  = dense_lower_diag_chain(6,  15, seed=3)
A_dense8  = dense_lower_diag_chain(8,  15, seed=5)
A_dense10 = dense_lower_diag_chain(10, 20, seed=9)

def full_dense(n, sigma, seed):
    r = np.random.default_rng(seed)
    while True:
        A = r.integers(-sigma, sigma+1, size=(n, n))
        if np.linalg.matrix_rank(A) == n and all(A[i, i] != 0 for i in range(n)):
            return A

A_full6 = full_dense(6, 10, seed=17)
A_full8 = full_dense(8, 12, seed=23)

run_and_plot(A_D1, K=0, sigma=int(np.max(np.abs(A_D1))),
             title="D1: 5x5 sparse", outpath="heatmap_D1.png")
run_and_plot(A_D2, K=0, sigma=int(np.max(np.abs(A_D2))),
             title="D2: 6x6 sparser", outpath="heatmap_D2.png")
run_and_plot(A_D3, K=1, sigma=int(np.max(np.abs(A_D3))),
             title="D3: 5x5 K=1", outpath="heatmap_D3.png")
run_and_plot(A_big, K=0, sigma=int(np.max(np.abs(A_big))),
             title="Big: 8x8 diag chain", outpath="heatmap_big.png")

run_and_plot(A_dense6, K=0, sigma=int(np.max(np.abs(A_dense6))),
             title="Dense6: 6x6 lower-tri", outpath="heatmap_dense6.png")
run_and_plot(A_dense8, K=0, sigma=int(np.max(np.abs(A_dense8))),
             title="Dense8: 8x8 lower-tri", outpath="heatmap_dense8.png")
run_and_plot(A_dense10, K=0, sigma=int(np.max(np.abs(A_dense10))),
             title="Dense10: 10x10 lower-tri", outpath="heatmap_dense10.png")

run_and_plot(A_full6, K=0, sigma=int(np.max(np.abs(A_full6))),
             title="Full6: 6x6 fully dense", outpath="heatmap_full6.png")
run_and_plot(A_full8, K=0, sigma=int(np.max(np.abs(A_full8))),
             title="Full8: 8x8 fully dense", outpath="heatmap_full8.png")

A_full12 = full_dense(12, 20, seed=41)
A_full16 = full_dense(16, 30, seed=57)
A_full20 = full_dense(20, 50, seed=71)

def conditioned_int_matrix(n, sigma, kappa, seed):
    """Random integer matrix with target condition number ~kappa and entries
    bounded roughly by sigma. Built as round(scale * U diag(s) V^T) with
    s spread log-linearly from 1 down to 1/kappa, then U,V are Haar-random
    orthogonal matrices from a QR of Gaussian."""
    r = np.random.default_rng(seed)
    for _ in range(200):
        G1 = r.standard_normal((n, n))
        G2 = r.standard_normal((n, n))
        U, _ = np.linalg.qr(G1)
        V, _ = np.linalg.qr(G2)
        s = np.logspace(0, -np.log10(kappa), n)
        M = U @ np.diag(s) @ V.T
        scale = sigma / np.max(np.abs(M))
        A = np.round(scale * M).astype(int)
        # ensure nonzero diagonal + full rank
        if np.linalg.matrix_rank(A) == n and all(A[i, i] != 0 for i in range(n)):
            return A
    raise RuntimeError("could not build conditioned matrix")

run_and_plot(A_full12, K=0, sigma=int(np.max(np.abs(A_full12))),
             title="Full12: 12x12 fully dense", outpath="heatmap_full12.png")
run_and_plot(A_full16, K=0, sigma=int(np.max(np.abs(A_full16))),
             title="Full16: 16x16 fully dense", outpath="heatmap_full16.png")
run_and_plot(A_full20, K=0, sigma=int(np.max(np.abs(A_full20))),
             title="Full20: 20x20 fully dense", outpath="heatmap_full20.png")

for kappa in [1e2, 1e4, 1e6, 1e8, 1e10]:
    A_k = conditioned_int_matrix(20, sigma=50, kappa=kappa, seed=int(kappa) % 997 + 1)
    cond = np.linalg.cond(A_k.astype(float))
    tag = f"kappa1e{int(np.log10(kappa))}"
    run_and_plot(A_k, K=0, sigma=int(np.max(np.abs(A_k))),
                 title=f"Cond20 target k={kappa:.0e}, actual={cond:.1e}",
                 outpath=f"heatmap_cond20_{tag}.png")

def sylvester_hadamard(k):
    """Return H_{2^k}: a 2^k x 2^k Hadamard matrix with entries in {+1, -1}.
    Rows are mutually orthogonal, so this saturates the Hadamard bound."""
    H = np.array([[1]])
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H

# H_16: n=16, sigma=1, bound = 16*log2(sqrt(16)) = 16*2 = 32. |det| = 16^8, log2 = 32. Saturating.
H16 = sylvester_hadamard(4)
# Add a random diagonal offset to break degenerate leading minors while
# keeping near-orthogonality. Small shift stays near the Hadamard bound.
rng = np.random.default_rng(2027)
H16_p = H16.copy()
for i in range(16):
    H16_p[i, i] += rng.integers(1, 4)  # +1..+3 on the diagonal
# Verify no zero pivot in Bareiss
def check_bareiss_ok(A):
    A = A.astype(object).copy()
    n = A.shape[0]
    prev = 1
    for k in range(n):
        if A[k, k] == 0:
            return False
        for i in range(k+1, n):
            for j in range(k+1, n):
                num = A[k, k]*A[i, j] - A[i, k]*A[k, j]
                A[i, j] = num // prev
        prev = A[k, k]
    return True

for attempt in range(50):
    if check_bareiss_ok(H16_p):
        break
    H16_p = H16.copy()
    for i in range(16):
        H16_p[i, i] += rng.integers(1, 4)
print(f"Hadamard16 perturbed sigma = {int(np.max(np.abs(H16_p)))}, cond = {np.linalg.cond(H16_p.astype(float)):.2f}")
run_and_plot(H16_p, K=0, sigma=int(np.max(np.abs(H16_p))),
             title=f"Hadamard16 (Sylvester + diag shift), cond={np.linalg.cond(H16_p.astype(float)):.2f}",
             outpath="heatmap_hadamard16.png")

# Scaled Hadamard: A = sigma * H + small perturbation. Entries near sigma
# everywhere, near-orthogonal rows. This should push bits toward the bound.
for sigma_scale in [5, 10, 30]:
    A_sh = sigma_scale * H16
    r2 = np.random.default_rng(9000 + sigma_scale)
    for _ in range(50):
        A_try = A_sh.copy()
        for i in range(16):
            A_try[i, i] += r2.integers(1, sigma_scale // 2 + 2)
        if check_bareiss_ok(A_try):
            A_sh = A_try; break
    cond = np.linalg.cond(A_sh.astype(float))
    sigma = int(np.max(np.abs(A_sh)))
    run_and_plot(A_sh, K=0, sigma=sigma,
                 title=f"ScaledHadamard16 sigma={sigma}, cond={cond:.2f}",
                 outpath=f"heatmap_scaledhad16_s{sigma_scale}.png")

# Near-orthogonal: kappa close to 1 should push bits toward the Hadamard bound.
for i, kappa in enumerate([1.0, 1.5, 2.0, 3.0, 5.0, 10.0]):
    A_o = conditioned_int_matrix(20, sigma=50, kappa=kappa, seed=1000 + i)
    cond = np.linalg.cond(A_o.astype(float))
    tag = f"orth_k{kappa:g}".replace(".", "p")
    run_and_plot(A_o, K=0, sigma=int(np.max(np.abs(A_o))),
                 title=f"NearOrth20 target k={kappa:g}, actual={cond:.2f}",
                 outpath=f"heatmap_{tag}.png")
