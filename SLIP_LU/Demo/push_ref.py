"""Reference implementation of the SDIP algorithm using sLTDpush


Everything is 0-indexed:
L[i][k] holds SLIP integer L entry for i >= k, with L[k][k] = rho[k].
U[k][j] holds SLIP integer U entry for k <= j, with U[k][k] = rho[k].
rho[k] = det of leading (k+1)x(k+1) principal minor of PAQ.
rho[-1] = 1.
"""

import sys


def read_triplet(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    m, n, nnz = int(next(it)), int(next(it)), int(next(it))
    A = [[0]*n for _ in range(m)]
    for _ in range(nnz):
        i = int(next(it)) - 1
        j = int(next(it)) - 1
        v = int(next(it))
        A[i][j] = v
    return A


def bareiss(A_in):
    """Fraction-free elimination, returns (L, U, rho)."""
    n = len(A_in)
    M = [row[:] for row in A_in]
    prev = 1
    for k in range(n):
        pivot = M[k][k]
        if pivot == 0:
            raise RuntimeError(f"Zero leading principal minor at k={k}")
        for i in range(k+1, n):
            for j in range(k+1, n):
                num = pivot * M[i][j] - M[i][k] * M[k][j]
                assert num % prev == 0
                M[i][j] = num // prev
        prev = pivot
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]
    rho = [0]*n
    for k in range(n):
        rho[k] = M[k][k]
        L[k][k] = M[k][k]
        U[k][k] = M[k][k]
        for i in range(k+1, n):
            L[i][k] = M[i][k]
        for j in range(k+1, n):
            U[k][j] = M[k][j]
    return L, U, rho


def cycle(A, K):
    n = len(A)
    perm = list(range(n))
    for i in range(K, n-1):
        perm[i] = i + 1
    perm[n-1] = K
    return [[A[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


def push_eager(L, U, rho, K):
    """Distant diagonal push of column K to column n-1.
    Eager version, no history vector. Returns (L_hat, U_hat, rho_prime)."""
    n = len(L)
    L_hat = [[0]*n for _ in range(n)]
    U_hat = [[0]*n for _ in range(n)]
    rho_prime = [0]*n

    def rp(idx):
        """Access the current-partial pivot at position idx: rho_prime for idx >= K
        (once finalized) or rho for idx < K. idx = -1 gives 1."""
        if idx < 0:
            return 1
        if idx < K:
            return rho[idx]
        return rho_prime[idx]

    # Frozen prefix on L
    for j in range(K):
        for i in range(K):
            L_hat[i][j] = L[i][j]
        for i in range(K, n-1):
            L_hat[i][j] = L[i+1][j]
        L_hat[n-1][j] = L[K][j]
    # Frozen prefix on U
    for i in range(K):
        for j in range(K):
            U_hat[i][j] = U[i][j]
        for j in range(K, n-1):
            U_hat[i][j] = U[i][j+1]
        U_hat[i][n-1] = U[i][K]
    for k in range(K):
        rho_prime[k] = rho[k]

    # Initialize spikes to the K-th column/row of L, U (unchanged from their Bareiss stage)
    mu_L = [L[i][K] for i in range(n)]
    mu_U = [U[K][j] for j in range(n)]

    for k in range(K, n-1):
        s_U = mu_U[k+1]
        s_L = mu_L[k+1]

        # Sylvester pivot update
        rho_km1_p = rp(k-1)
        num = rho_km1_p * rho[k+1] + s_L * s_U
        if num == 0:
            raise RuntimeError(f"Zero pivot at k={k}")
        assert num % rho[k] == 0
        rho_prime[k] = num // rho[k]

        # --- Backtracking on L: write L_hat[:, k] ---
        # For i in {k+2..n-1}, write L_hat[i-1, k].
        # Bottom row: L_hat[n-1, k] = s_U (the corner).
        for i in range(k+2, n):
            num = rho_km1_p * L[i][k+1] + s_U * mu_L[i]
            assert num % rho[k] == 0
            L_hat[i-1][k] = num // rho[k]
        L_hat[n-1][k] = s_U

        # --- Backtracking on U^T (write U_hat row k): symmetric ---
        for j in range(k+2, n):
            num = rho_km1_p * U[k+1][j] + s_L * mu_U[j]
            assert num % rho[k] == 0
            U_hat[k][j-1] = num // rho[k]
        U_hat[k][n-1] = s_L

        # --- RwSOP: update spike mu_L for next iteration ---
        # μ_i^new = (ρ'_k · μ_i^old − s_L · L̂[i-1, k]) / ρ'_{k-1}, for i in {k+2..n-1}
        new_mu_L = [0]*n
        for i in range(k+2, n):
            num = rho_prime[k] * mu_L[i] - s_L * L_hat[i-1][k]
            assert num % rho_km1_p == 0, (
                f"mu_L[{i}] RwSOP fail k={k}: {num} / {rho_km1_p}")
            new_mu_L[i] = num // rho_km1_p
        for i in range(k+2, n):
            mu_L[i] = new_mu_L[i]
        mu_L[k+1] = 0  # corner consumed

        # --- RwSOP on U (symmetric with s_U as multiplier, U_hat[k, j-1] as read) ---
        new_mu_U = [0]*n
        for j in range(k+2, n):
            num = rho_prime[k] * mu_U[j] - s_U * U_hat[k][j-1]
            assert num % rho_km1_p == 0
            new_mu_U[j] = num // rho_km1_p
        for j in range(k+2, n):
            mu_U[j] = new_mu_U[j]
        mu_U[k+1] = 0

    # After the chain, the spike is at position n-1.
    # μ_L values at rows k+2..n-1 (well, valid k+2 for the last iteration k=n-2 is n,
    # so no more RwSOP updates were done; mu_L is at whatever it was at end of iter n-3).
    #
    # Actually, at the end of iteration k=n-2:
    #   Backtracking wrote L_hat[:, n-2]: rows k+1..n-2 via loop, row n-1 = s_U.
    #   RwSOP loop was range(n, n) — empty. So mu_L was not updated in that iteration.
    #   Meaning mu_L is still at its state from the end of iteration k=n-3's RwSOP.
    #
    # We need L_hat[:, n-1] and the pivot ρ'_{n-1}. What are they?
    # For a cyclic shift, L_hat[:, n-1] is the last column of L_new, and ρ'_{n-1} is
    # det(A_new) which equals det(A_orig) * (-1)^(n-1-K).
    #
    # Simplest correct approach: run one more "phantom" step or fill via mu directly.
    # At end of chain, the spike μ_L should represent "column K after being pushed to
    # column n-1". In the swapped-in-place perm.c convention, after all n-K-1 pushes,
    # the pushed column ends up at position n-1, and its values are L_new[:, n-1].
    #
    # But our mu_L is UPDATED THROUGH the chain, so at end it should already equal
    # L_new[:, n-1]... let me check.

    # Last column of L_hat and last row of U_hat are trivial: only the diagonal is
    # nonzero, and equal to rho_prime[n-1]. For a symmetric row+col cyclic shift,
    # det(A_new) = det(A_orig), so rho_prime[n-1] = rho[n-1].
    rho_prime[n-1] = rho[n-1]

    # Diagonal: L_hat[k][k] = U_hat[k][k] = rho_prime[k]
    for k in range(n):
        L_hat[k][k] = rho_prime[k]
        U_hat[k][k] = rho_prime[k]

    return L_hat, U_hat, rho_prime


def push(L, U, rho, K):
    """Distant diagonal push of column K to column n-1, WITH history-vector
    deferred scaling on the spike μ. Returns (L_hat, U_hat, rho_prime).

    The invariant: h_L[i] is the chain-stage index that μ_L[i] was last
    eagerly updated to. The true "current-partial stage-t" value of μ_L[i]
    (for any t >= h_L[i], assuming no firing updates were skipped in between)
    equals μ_L[i] · rp(t) / rp(h_L[i]). This lets us skip mpz work for rows
    that no firing update touches at a given iteration.
    """
    n = len(L)
    L_hat = [[0]*n for _ in range(n)]
    U_hat = [[0]*n for _ in range(n)]
    rho_prime = [0]*n

    # touch counters just to measure the deferral savings vs. eager
    bring_up_count = [0]

    def rp(idx):
        if idx < 0:
            return 1
        if idx < K:
            return rho[idx]
        return rho_prime[idx]

    def bring_up(mu, h, i, target):
        """Bring mu[i] up to chain stage 'target' via history scaling.
        No-op if already at or beyond target, or if mu[i] is zero."""
        if h[i] >= target:
            return
        if mu[i] == 0:
            h[i] = target
            return
        src = rp(h[i])
        dst = rp(target)
        new_val = dst * mu[i]
        assert new_val % src == 0, (
            f"bring_up mu[{i}] {h[i]}->{target}: {new_val}/{src}")
        mu[i] = new_val // src
        h[i] = target
        bring_up_count[0] += 1

    # Frozen prefix (identical to eager version)
    for j in range(K):
        for i in range(K):
            L_hat[i][j] = L[i][j]
        for i in range(K, n-1):
            L_hat[i][j] = L[i+1][j]
        L_hat[n-1][j] = L[K][j]
    for i in range(K):
        for j in range(K):
            U_hat[i][j] = U[i][j]
        for j in range(K, n-1):
            U_hat[i][j] = U[i][j+1]
        U_hat[i][n-1] = U[i][K]
    for k in range(K):
        rho_prime[k] = rho[k]

    # Initial spike, initial history at K-1 (so the first iteration k=K bring-up
    # to k-1 = K-1 is a no-op, matching the eager version).
    mu_L = [L[i][K] for i in range(n)]
    mu_U = [U[K][j] for j in range(n)]
    h_L = [K - 1] * n
    h_U = [K - 1] * n

    for k in range(K, n-1):
        rho_km1_p = rp(k-1)

        # Corner: bring up so the Sylvester numerator is at consistent stages.
        bring_up(mu_L, h_L, k+1, k-1)
        bring_up(mu_U, h_U, k+1, k-1)
        s_L = mu_L[k+1]
        s_U = mu_U[k+1]

        num = rho_km1_p * rho[k+1] + s_L * s_U
        if num == 0:
            raise RuntimeError(f"Zero pivot at k={k}")
        assert num % rho[k] == 0
        rho_prime[k] = num // rho[k]

        # --- Backtracking L: fire iff L[i, k+1] != 0 OR mu_L[i] != 0 ---
        for i in range(k+2, n):
            if L[i][k+1] == 0 and mu_L[i] == 0:
                continue
            bring_up(mu_L, h_L, i, k-1)
            num = rho_km1_p * L[i][k+1] + s_U * mu_L[i]
            assert num % rho[k] == 0
            L_hat[i-1][k] = num // rho[k]
        L_hat[n-1][k] = s_U

        # --- Backtracking U: symmetric ---
        for j in range(k+2, n):
            if U[k+1][j] == 0 and mu_U[j] == 0:
                continue
            bring_up(mu_U, h_U, j, k-1)
            num = rho_km1_p * U[k+1][j] + s_L * mu_U[j]
            assert num % rho[k] == 0
            U_hat[k][j-1] = num // rho[k]
        U_hat[k][n-1] = s_L

        # --- RwSOP L: fire iff L_hat[i-1, k] != 0. Defer pure scalings. ---
        for i in range(k+2, n):
            if L_hat[i-1][k] == 0:
                continue
            bring_up(mu_L, h_L, i, k-1)
            num = rho_prime[k] * mu_L[i] - s_L * L_hat[i-1][k]
            assert num % rho_km1_p == 0
            mu_L[i] = num // rho_km1_p
            h_L[i] = k
        mu_L[k+1] = 0
        h_L[k+1] = k

        # --- RwSOP U: symmetric ---
        for j in range(k+2, n):
            if U_hat[k][j-1] == 0:
                continue
            bring_up(mu_U, h_U, j, k-1)
            num = rho_prime[k] * mu_U[j] - s_U * U_hat[k][j-1]
            assert num % rho_km1_p == 0
            mu_U[j] = num // rho_km1_p
            h_U[j] = k
        mu_U[k+1] = 0
        h_U[k+1] = k

    rho_prime[n-1] = rho[n-1]
    for k in range(n):
        L_hat[k][k] = rho_prime[k]
        U_hat[k][k] = rho_prime[k]

    return L_hat, U_hat, rho_prime, bring_up_count[0]


def matdiff(A, B, label):
    n = len(A)
    fails = []
    for i in range(n):
        for j in range(n):
            if A[i][j] != B[i][j]:
                fails.append((i, j, A[i][j], B[i][j]))
    if fails:
        print(f"  {label}: MISMATCH ({len(fails)} entries)")
        for (i, j, a, b) in fails[:8]:
            print(f"    [{i}][{j}]: got {a}, want {b}")
        return False
    print(f"  {label}: OK")
    return True


def matequal(A, B):
    return all(A[i][j] == B[i][j] for i in range(len(A)) for j in range(len(A)))


def run(path):
    A = read_triplet(path)
    n = len(A)
    print(f"Loaded {path}, n={n}")
    L, U, rho = bareiss(A)

    all_ok = True
    for K in range(n - 1):
        A_new = cycle(A, K)
        Lx, Ux, rx = bareiss(A_new)
        try:
            Le, Ue, re_ = push_eager(L, U, rho, K)
            Lh, Uh, rh, bu = push(L, U, rho, K)
            # eager upper bound on bring-ups: 2 * (n - K - 1) * n
            eager_bound = 2 * max(0, n - K - 1) * n
        except Exception as e:
            print(f"K={K}: push RAISED: {e}")
            all_ok = False
            continue
        ok_ref = (rh == rx) and matequal(Lh, Lx) and matequal(Uh, Ux)
        ok_eq  = (re_ == rh) and matequal(Le, Lh) and matequal(Ue, Uh)
        print(f"K={K}: history vs refactor = {'OK' if ok_ref else 'MISMATCH'}, "
              f"history vs eager = {'OK' if ok_eq else 'MISMATCH'}, "
              f"bring-ups = {bu} / eager {eager_bound}")
        if not (ok_ref and ok_eq):
            all_ok = False
            if not matequal(Lh, Lx):
                matdiff(Lh, Lx, "L_hat (history vs refactor)")
            if not matequal(Uh, Ux):
                matdiff(Uh, Ux, "U_hat (history vs refactor)")
            if not matequal(Le, Lh):
                matdiff(Le, Lh, "L_hat (eager vs history)")
    print()
    print("ALL PASS" if all_ok else "FAILURES ABOVE")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dd8.txt"
    run(path)
