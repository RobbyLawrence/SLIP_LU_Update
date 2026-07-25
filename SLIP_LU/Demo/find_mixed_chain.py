"""Scan small random matrices for ones that force a column push in the sLTDpush chain.

push_ref.py's push() raises "Zero pivot at k=..." exactly when the diagonal-push
Sylvester numerator hits zero. Any (matrix, K) pair that raises is a mixed-chain
test case for sdip.c's column-push fallback.

Prints matrices in SLIP triplet format so they can be dropped into a .txt file
and fed straight to ./bin/sdip.
"""

import random
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("push_ref", "push_ref.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def try_matrix(A):
    """Return list of (K, failing_k) pairs, or []."""
    n = len(A)
    try:
        L, U, rho = pr.bareiss(A)
    except RuntimeError:
        return None
    hits = []
    for K in range(n - 1):
        try:
            pr.push(L, U, rho, K)
        except RuntimeError as e:
            if "Zero pivot at k=" in str(e):
                fail_k = int(str(e).split("k=")[1])
                hits.append((K, fail_k))
    return hits


def random_matrix(n, lo=-3, hi=3, density=1.0, seed=None):
    if seed is not None:
        random.seed(seed)
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() < density:
                v = 0
                while v == 0:
                    v = random.randint(lo, hi)
                A[i][j] = v
    return A


def fmt_slip_triplet(A):
    """Emit A in SLIP triplet format (1-indexed)."""
    n = len(A)
    nnz = sum(1 for i in range(n) for j in range(n) if A[i][j] != 0)
    lines = [f"{n} {n} {nnz}"]
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0:
                lines.append(f"{i+1} {j+1} {A[i][j]}")
    return "\n".join(lines)


def main():
    # Find cases where col push is NOT the first step of the chain
    for n in (5, 6, 7, 8):
        print(f"### scanning n={n} (fail_k > K, i.e., mixed after diagonals) ###")
        found = 0
        for trial in range(30000):
            A = random_matrix(n, lo=-3, hi=3, density=0.9, seed=(n*1000 + trial))
            hits = try_matrix(A)
            if not hits: continue
            interesting = [(K, fk) for (K, fk) in hits if fk > K]
            if interesting:
                K, fk = interesting[0]
                print(f"  n={n} trial={trial}: K={K} col push at k={fk} (all hits={hits})")
                print(fmt_slip_triplet(A))
                print()
                found += 1
                if found >= 3: break
        if found == 0:
            print(f"  none in {trial+1} trials")


if __name__ == "__main__":
    main()
