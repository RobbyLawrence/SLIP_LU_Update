# Column-Replacement Update for SLIP LU — Implementation Spec

**Audience:** an agent implementing this in the SLIP LU (Sparse Left-looking
Integer-Preserving LU) C codebase, then benchmarking it against full
refactorization.

**One-line summary:** A column replacement at position `p` leaves factor
columns `1..p-1` untouched; recompute only columns `p..n`, reusing the existing
sparse REF triangular solve. Implement it as the existing factorization loop
*warm-started* from the preserved prefix. Optionally skip trailing columns that
are provably unaffected.

---

## 1. Background invariant (do not violate)

The factorization maintains, in exact integer arithmetic (GMP `mpz_t`):

```
P * B * Q = L * D^{-1} * U,    D = diag(rho_{k-1} * rho_k)^{-1},   rho_0 = 1
```

- `L` (lower) and `U` (upper) are stored CSC with `mpz_t` values. Their
  diagonals are the pivots `rho_k`. Off-diagonal entries are subdeterminants of
  the (row-permuted) input — their *values depend on the pivot sequence*, not
  just on storage position. This is why a column reorder cannot be done by
  relabeling alone; affected columns must be recomputed.
- `rho_1..rho_n` are the ordered pivots; `rho_n = ±det(B)`.
- `Q` is the fixed COLAMD column preorder. `P` (equivalently `pinv`) is the row
  permutation chosen by partial pivoting.
- Pivoting rule: at each column, the pivot is the **smallest-magnitude**
  eligible entry (this bounds entry bit-length via Hadamard). Preserve this rule
  exactly so bit-lengths stay within the original factorization's bound.

Every entry stays integral and bounded by the SLIP/Hadamard bit-length bound.
The update must not grow bit-length or introduce fill beyond what the
elimination structure forces.

---

## 2. Orient yourself in this codebase first (Step 0)

Before writing code, locate and bind these concepts to the actual symbols.
Names below are likely (`SLIP_*` or `SPEX_*`); **confirm against the headers**.

| Concept | Likely symbol(s) — verify | Notes |
|---|---|---|
| Sparse matrix struct (CSC, mpz) | `SLIP_matrix` / `SPEX_matrix` | fields like `m,n,nz,p,i,x.mpz` |
| Factorization driver | `slip_left_lu_factorize` / `SPEX_Left_LU_factorize` | the per-column loop is the template for the update |
| Sparse REF triangular solve (Algorithm 1) | `slip_ref_triangular_solve` | computes one column of L and U; **reuse this verbatim** |
| Symbolic reach / DFS | `slip_reach`, `slip_dfs` | gives `X = Reach_{G_L}(b)` in topological order |
| Pivot array `rho` | `rhos` (dense mpz vector) | confirm 0- vs 1-based indexing |
| Row permutation | `pinv` / `row_perm` | |
| Column order `Q` | `S->q` from symbolic analysis | fixed COLAMD order |
| Smallest-pivot selection | `slip_get_pivot` / `..._smallest_pivot` | keep identical rule |
| History vector | local `int *h` inside the solve | one entry per row; reset per column |

**Action:** read the body of the factorization driver and the triangular solve
end-to-end. The update is a specialization of that driver; matching its exact
calling conventions (how `A(:,k)` is fed in, how `L/U/rhos/pinv` are written,
how `h` and the workspace are reset) is the single most important thing for
correctness.

---

## 3. The algorithm

`Reach_{G_L}` = depth-first reachability in the graph of `L` (edge `j -> i` iff
`L(i,j) != 0`), exactly as the factorization's symbolic phase. The numeric
solve is SLIP **Algorithm 1** (restated in §4 for reference).

```
SLIP_COLUMN_REPLACE(L, U, rhos, pinv, Q, c, a_new):
  # Replace original column c of B with a_new. Preserve the §1 invariant.

  p <- position such that Q[p] == c          # column's slot in the factor order
  A' <- A with column c replaced by a_new     # only this column differs

  # ---- Phase 0: keep the unaffected prefix ----
  # L(:,1..p-1), U(:,1..p-1), rhos[1..p-1], and pinv for positions 1..p-1
  # are identical to the old factors. Seed the new factors with them.
  Delta         <- {p}        # pivot columns whose L-content changed
  prefix_stable <- true       # no pivot-row reassignment has happened yet

  # ---- Phase 1+2: left-looking re-solve of positions p..n ----
  for k = p to n:
     b_k <- A'(:, Q[k])        # = a_new when k==p; unchanged raw column otherwise

     # symbolic short-circuit (optional optimization; see milestone B)
     if k > p and prefix_stable:
        X_k <- Reach_{G_L}(b_k)               # over current first k-1 cols of L
        if X_k ∩ Delta == empty:
            copy old column k into the new factors; continue   # ZERO numeric work

     # numeric: SLIP Algorithm 1 against the CURRENT L^{(k-1)}, rhos
     (x, X_k) <- SPARSE_REF_TRIANGULAR_SOLVE(L_partial, rhos, b_k)

     # partial pivoting: smallest-magnitude eligible (un-pivoted) entry of the L-part
     r <- argmin{ |x_i| : i un-pivoted, x_i != 0 }
     if r != old_pivot_row(k):
        prefix_stable <- false                # forces recompute of all later cols
     set pinv / row_perm for position k -> r

     write U(:,k) <- { x_j : j already pivoted }          # upper part
     write L(:,k) <- { x_i : i un-pivoted }, with rho_k = x_r on the diagonal
     rhos[k] <- x_r
     if (X_k, x) differ from the stored column k:  Delta <- Delta ∪ {k}

  # ---- Phase 3: validate singularity ----
  if rhos[n] == 0: return SINGULAR            # B' is singular
  return OK
```

### Two correctness subtleties (must handle)

1. **Structural skip is exact.** If `b_k`'s reach avoids every changed column,
   the solve touches only unchanged `L` columns and unchanged raw data, so the
   result is bit-identical to the stored column. The skip is safe **only** while
   `prefix_stable` holds (see next point).

2. **Pivoting is globally coupled.** When position `p` (or any recomputed
   column) selects a *different* pivot row than before, the set of eligible rows
   for later positions shifts, which can change a later pivot even for a
   structurally independent column. The conservative, correct rule: once any
   pivot row is reassigned, set `prefix_stable = false` and recompute **all**
   remaining columns. Implement this flag first. A finer version (skip column
   `k` iff `X_k` avoids `Delta` *and* the set of freed/consumed rows) is a later
   optimization — do not start with it.

---

## 4. Algorithm 1 (sparse REF triangular solve) — for reference

This already exists in the codebase; reuse it. Restated so you can confirm the
version you call matches. Indices are 1-based with `rho_0 = 1`; align to the
code's actual indexing.

```
x = b
X = Reach_{G_L}(b)            # DFS, topological order
sort(X)
h_j = 0  for all j in X       # history vector
for j in X (sorted ascending):
    if h_j < min(j,k) - 1:
        x_j = x_j * rho_{min(j,k)-1} / rho_{h_j}       # history update (exact)
    for each i > j with L(i,j) != 0:
        if h_i < j - 1:
            x_i = x_i * rho_{j-1} / rho_{h_i}          # history update (exact)
        x_i = (rho_j * x_i - L(i,j) * x_j) / rho_{j-1} # IPGE update (exact)
        h_i = j
# output: x_j = U(j,k) for j<k ; rho_k = x_k ; L(j,k) = x_j for j>k
```

Every `/` here is an exact integer division by the Bareiss/Sylvester identity.
Keep divisions as the **last** operation (`mpz_divexact`) and assert zero
remainder in debug builds.

---

## 5. Recommended implementation strategy — warm-started factorize loop

Do **not** try to mutate `L`/`U` CSC in place (different column nnz makes that
painful). Instead, build the new `L`/`U` the same way the driver does, but
pre-load the first `p-1` columns:

1. **Clone the factorization driver** into a new function (§6).
2. **Seed the partial state at step `p`** by copying from the old factors:
   - `L`'s columns `1..p-1` (CSC: copy `p[]`, `i[]`, `x.mpz[]` prefixes),
   - `U`'s columns `1..p-1`,
   - `rhos[1..p-1]`,
   - `pinv` / `row_perm` for positions `1..p-1`,
   - the symbolic graph `G_L` restricted to those columns (whatever structure
     the reach routine consumes).
3. **Run the loop from `k = p`**, feeding `a_new` at `k=p` and `A(:,Q[k])`
   otherwise. This is byte-for-byte the work the original factorization did for
   `k >= p`, so the result is a valid SLIP LU of `B'`.
4. Append computed columns exactly as the driver appends them (so CSC stays
   well-formed and you never overwrite a live column).
5. Add the §3 skip/`prefix_stable` logic around the per-column call.

This reuse maximizes correctness: the only new logic is prefix seeding + the
skip guard. Everything numeric goes through the already-validated solve and
pivot routines.

**Column mapping:** the application says "replace column `c`." Find `p` with
`Q[p] == c`. For `k > p`, raw columns are unchanged (`A(:,Q[k])`); only the
`k=p` raw column becomes `a_new`. Reuse `Q` as-is (a single column change does
not warrant recomputing COLAMD).

---

## 6. Suggested function signature & files

Add to the factorization source/header (match existing style, error codes, and
`SLIP_info`/`SPEX_info` return convention):

```c
/* Update the SLIP LU factorization (L, U, rhos, pinv) of B to that of B',
 * where B' equals B with original column `col` replaced by `a_new`.
 * Q (column order) is reused. On singular B', returns the library's
 * singular-matrix code and leaves outputs unspecified (caller refactorizes).
 */
SLIP_info slip_lu_update_column_replace
(
    SLIP_matrix **L_handle,      /* in/out: lower factor (rebuilt)            */
    SLIP_matrix **U_handle,      /* in/out: upper factor (rebuilt)            */
    SLIP_matrix  *rhos,          /* in/out: pivot sequence                    */
    int64_t      *pinv,          /* in/out: row permutation                   */
    const int64_t *Q,            /* in: fixed column order (COLAMD)           */
    SLIP_matrix  *A,             /* in: original matrix (for unchanged cols)  */
    int64_t       col,           /* in: original column index being replaced  */
    const SLIP_matrix *a_new,    /* in: replacement column (n x 1, mpz)       */
    const SLIP_options *option
);
```

Adjust types to the codebase (handles vs values, `int64_t` vs `int32_t`,
`SLIP` vs `SPEX`). Return the existing singular code on `rhos[n] == 0`.

---

## 7. Memory / mpz hygiene

- Every `mpz_t` written must be `mpz_init`'d once and `mpz_clear`'d exactly once.
  Mirror the driver's allocation pattern for `L`/`U`/`rhos`.
- Reuse the driver's dense `mpz_t` workspace and integer `h`/marker arrays;
  reset them per column exactly as the driver does.
- On any error path, free partially built factors to avoid leaks (follow the
  existing `SLIP_FREE_ALL` / `SPEX_FREE_ALL` macro convention).
- Prefer `mpz_divexact` over `mpz_tdiv_q` for the REF divisions.

---

## 8. Correctness invariants & runtime assertions (debug builds)

Add these as asserts gated behind a debug flag (these match the project's
existing diagnostics):

- **Exact division:** after each REF division, remainder is zero
  (`mpz_divisible_p` or a `tdiv_qr` remainder check).
- **Bit-length ceiling:** every produced entry satisfies the per-entry Hadamard
  bit bound; assert `mpz_sizeinbase(x,2) <= bound`.
- **Pivot nonzero:** `rhos[k] != 0` for `k <= n-1` during the loop; final
  `rhos[n] != 0` (else singular).
- **Determinant:** `rhos[n] == ±det(B')` (cross-check against an independent det
  for small instances).

---

## 9. Validation oracle (vs refactorization) — the acceptance test

Two independent checks. Implement **both**; (a) is the ground truth.

**(a) Reconstruct and compare exactly.** Verify `P * B' * Q == L * D^{-1} * U`
by an exact multiply-back: equivalently check that `L * D^{-1} * U` applied to
each unit column reproduces `P B' Q`, or reuse the library's
`SLIP_check`/reconstruction routine on the updated factors. This must hold
bit-exactly regardless of pivot tie-breaks.

**(b) Compare to a fresh factorization.** Factorize `B'` from scratch with the
**same** `Q` and the **same** pivoting rule and tie-break. If the tie-break is
deterministic, `L`, `U`, `rhos`, `pinv` should match the update's output
entry-for-entry. If they differ, first confirm it is only a pivot tie (same
`|x_i|`); a non-tie difference is a bug.

**(c) Solve check.** Pick a random integer RHS `b`; solve `B' x = b` exactly via
the updated factors and via the fresh factors; assert the exact rational
solutions are equal and that `B' x == b`.

Run (a)+(c) on every test instance; run (b) where tie-breaks are deterministic.

---

## 10. Benchmark harness (update vs refactorization)

Use the QSopt_ex / BasisLIB_INT instances already wired into the project.

For each instance:
1. Build an initial nonsingular basis `B` (m x m) and factorize it (SLIP LU).
2. Generate a sequence of column replacements that mimic simplex basis changes:
   pick an existing column to leave and a new column (e.g., another column of
   the constraint matrix) to enter; ensure `B'` stays nonsingular (skip/regen if
   `rhos[n] == 0`).
3. For each replacement, time **both**:
   - `slip_lu_update_column_replace`, and
   - a full `slip_left_lu_factorize` of `B'`.
   using the project's existing timer.
4. After each, run the §9(a) reconstruction check.

**Report per replacement and aggregated (geometric mean):**
- wall-clock: update vs refactorize, and speedup ratio;
- `position p` of the replaced column (expect speedup to rise as `p -> n`);
- size of the recomputed/affected set `|{k >= p : recomputed}|`;
- `nnz(L) + nnz(U)` before vs after (fill growth);
- max entry bit-length before vs after (must not exceed the bound);
- number of pivot-row reassignments (how often `prefix_stable` trips).

Expected shape of results: update wins by a wide margin for late/structurally
local replacements; the two converge (update can lose) for early replacements
that reach most of the trailing factor — that crossover is the headline finding
to characterize.

---

## 11. Build milestones (implement and validate in this order)

- **A — Correct baseline.** Warm-started loop, **no skipping**: recompute every
  column `p..n` unconditionally. Pass §9(a) and §9(c) on all instances. This is
  the reference implementation and the oracle for later versions.
- **B — Symbolic skip.** Add `Reach`/`Delta`/`prefix_stable` to skip unaffected
  trailing columns. Output must remain bit-identical to milestone A (assert it).
- **C — Benchmark.** Wire in §10; produce the table and the crossover plot.
- **D (optional) — Finer skip.** Replace the global `prefix_stable` flag with
  freed/consumed-row tracking. Must stay bit-identical to A.

Get A green before touching B. B must equal A. C measures. D is gravy.

---

## 12. Out of scope (future optimization, note only)

A sparse port of Escobedo's push-and-swap (REF CRU) could avoid recomputing
whole trailing columns by propagating a local correction (`O(n^2)`-style dense
behavior instead of partial refactorization). It is deliberately **not** this
task because:
- its per-step operations (backtrack, row-wise switch of originating pivot, sign
  flips) touch two-column bands of the trailing submatrix and can introduce fill
  that the re-solve never creates;
- it needs **row access to `U`** (store `U` in CSR as well, as Reid/Suhl do),
  which this baseline does not require.

The milestone-A re-solve is also the validation oracle for any future
push-and-swap implementation: that implementation must reproduce
`(L, U, rhos)` up to pivot ties.
