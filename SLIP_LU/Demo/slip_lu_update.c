//------------------------------------------------------------------------------
// SLIP_LU/Demo/slip_lu_update.c: column-replacement update for SLIP LU
//------------------------------------------------------------------------------

// Warm-started SLIP LU factorization driver that reuses the unaffected prefix
// of L, U, and rhos for column positions 0..p-1 (where p = position of the
// replaced column in the COLAMD order), then recomputes columns p..n-1 with
// the existing sparse REF triangular solve and pivot routines.
//
// Milestone A: no symbolic skip — recomputes every column k = p..n-1.
//
// Implementation strategy follows instructions.md §5: clone the body of
// SLIP_LU_factorize but pre-load the partial state from the existing factors.
// Row indices in the stored L and U have already been permuted by pinv at the
// end of factorize, so the prefix is un-permuted (via row_perm) before the
// loop begins, and the final permutation pass is repeated on completion.

#define UPDATE_FREE_WORK            \
    SLIP_matrix_free(&x, NULL);     \
    SLIP_FREE(xi);                  \
    SLIP_FREE(h);                   \
    SLIP_FREE(pivs);                \
    SLIP_FREE(row_perm);

#define SLIP_FREE_ALL               \
    UPDATE_FREE_WORK                \
    SLIP_matrix_free(&L, NULL);     \
    SLIP_matrix_free(&U, NULL);

#include "slip_lu_update.h"
#include "slip_internal.h"
#include <math.h>

SLIP_info slip_lu_update_column_replace
(
    SLIP_matrix **L_handle,
    SLIP_matrix **U_handle,
    SLIP_matrix  *rhos,
    int64_t      *pinv,
    const SLIP_LU_analysis *S,
    const SLIP_matrix *A_new,
    int64_t       col,
    const SLIP_options *option
)
{
    if (!slip_initialized()) return SLIP_PANIC;
    if (!L_handle || !U_handle || !*L_handle || !*U_handle ||
        !rhos || !pinv || !S || !A_new)
    {
        return SLIP_INCORRECT_INPUT;
    }
    SLIP_REQUIRE(A_new, SLIP_CSC, SLIP_MPZ);

    SLIP_matrix *L_old = *L_handle;
    SLIP_matrix *U_old = *U_handle;
    SLIP_REQUIRE(L_old, SLIP_CSC, SLIP_MPZ);
    SLIP_REQUIRE(U_old, SLIP_CSC, SLIP_MPZ);

    int64_t n = A_new->n;
    if (L_old->n != n || U_old->n != n) return SLIP_INCORRECT_INPUT;

    SLIP_info info;
    SLIP_matrix *L = NULL, *U = NULL, *x = NULL;
    int64_t *xi = NULL, *h = NULL, *pivs = NULL, *row_perm = NULL;

    //--------------------------------------------------------------------------
    // locate position p such that S->q[p] == col
    //--------------------------------------------------------------------------
    int64_t p_pos = -1;
    for (int64_t k = 0; k < n; k++)
    {
        if (S->q[k] == col) { p_pos = k; break; }
    }
    if (p_pos < 0) return SLIP_INCORRECT_INPUT;

    //--------------------------------------------------------------------------
    // build row_perm (inverse of pinv) and seed pivs from the prefix pivots
    //--------------------------------------------------------------------------
    row_perm = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    pivs     = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    h        = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    xi       = (int64_t *) SLIP_malloc(2 * n * sizeof(int64_t));
    if (!row_perm || !pivs || !h || !xi)
    {
        SLIP_FREE_ALL;
        return SLIP_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < n; i++)
    {
        row_perm[pinv[i]] = i;
        pivs[i] = -1;
        h[i] = -1;
    }
    // mark the first p_pos pivots as already taken
    for (int64_t k = 0; k < p_pos; k++)
    {
        pivs[row_perm[k]] = 1;
    }

    //--------------------------------------------------------------------------
    // allocate new L and U; copy unpermuted prefix from L_old, U_old
    //--------------------------------------------------------------------------
    int64_t lnz_prefix = L_old->p[p_pos];
    int64_t unz_prefix = U_old->p[p_pos];
    int64_t L_size = SLIP_MAX(L_old->nzmax, lnz_prefix + n);
    int64_t U_size = SLIP_MAX(U_old->nzmax, unz_prefix + n);

    SLIP_CHECK(SLIP_matrix_allocate(&L, SLIP_CSC, SLIP_MPZ, n, n, L_size,
        false, false, option));
    SLIP_CHECK(SLIP_matrix_allocate(&U, SLIP_CSC, SLIP_MPZ, n, n, U_size,
        false, false, option));

    // copy prefix column pointers; tail set to prefix nnz (empty cols)
    for (int64_t k = 0; k <= p_pos; k++) L->p[k] = L_old->p[k];
    for (int64_t k = p_pos + 1; k <= n; k++) L->p[k] = lnz_prefix;
    for (int64_t k = 0; k <= p_pos; k++) U->p[k] = U_old->p[k];
    for (int64_t k = p_pos + 1; k <= n; k++) U->p[k] = unz_prefix;

    // copy prefix entries, un-permuting row indices: original = row_perm[stored]
    for (int64_t m = 0; m < lnz_prefix; m++)
    {
        L->i[m] = row_perm[L_old->i[m]];
        size_t size;
        SLIP_CHECK(SLIP_mpz_sizeinbase(&size, L_old->x.mpz[m], 2));
        SLIP_CHECK(SLIP_mpz_init2(L->x.mpz[m], size + 2));
        SLIP_CHECK(SLIP_mpz_set(L->x.mpz[m], L_old->x.mpz[m]));
    }
    for (int64_t m = 0; m < unz_prefix; m++)
    {
        U->i[m] = row_perm[U_old->i[m]];
        size_t size;
        SLIP_CHECK(SLIP_mpz_sizeinbase(&size, U_old->x.mpz[m], 2));
        SLIP_CHECK(SLIP_mpz_init2(U->x.mpz[m], size + 2));
        SLIP_CHECK(SLIP_mpz_set(U->x.mpz[m], U_old->x.mpz[m]));
    }

    //--------------------------------------------------------------------------
    // allocate dense workspace x
    //--------------------------------------------------------------------------
    int64_t estimate = 64 * SLIP_MAX(2, (int64_t) ceil(log2((double) n)));
    SLIP_CHECK(SLIP_matrix_allocate(&x, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, false, option));
    for (int64_t i = 0; i < n; i++)
    {
        SLIP_CHECK(SLIP_mpz_init2(x->x.mpz[i], estimate));
    }

    //--------------------------------------------------------------------------
    // Main loop: k = p_pos .. n-1
    //--------------------------------------------------------------------------
    int64_t lnz = lnz_prefix, unz = unz_prefix;
    for (int64_t k = p_pos; k < n; k++)
    {
        int64_t top, pivot;
        L->p[k] = lnz;
        U->p[k] = unz;

        if (lnz + n > L->nzmax) SLIP_CHECK(slip_sparse_realloc(L));
        if (unz + n > U->nzmax) SLIP_CHECK(slip_sparse_realloc(U));

        // ensure tail column pointers stay non-negative for the DFS marker
        // dance (slip_dfs reads L->p[jnew] for jnew in [k, n-1] and expects an
        // empty range, which the trailing pointers provide).
        for (int64_t kk = k + 1; kk <= n; kk++) L->p[kk] = lnz;
        for (int64_t kk = k + 1; kk <= n; kk++) U->p[kk] = unz;

        SLIP_CHECK(slip_ref_triangular_solve(&top, L, A_new, k, xi,
            (const int64_t *) S->q, rhos,
            (const int64_t *) pinv, (const int64_t *) row_perm, h, x));

        SLIP_CHECK(slip_get_pivot(&pivot, x, pivs, n, top, xi,
            S->q[k], k, rhos, pinv, row_perm, option));

        for (int64_t j = top; j < n; j++)
        {
            int64_t jnew = xi[j];
            int64_t loc  = pinv[jnew];
            size_t size;
            if (loc <= k)
            {
                U->i[unz] = jnew;
                SLIP_CHECK(SLIP_mpz_sizeinbase(&size, x->x.mpz[jnew], 2));
                SLIP_CHECK(SLIP_mpz_init2(U->x.mpz[unz], size + 2));
                SLIP_CHECK(SLIP_mpz_set(U->x.mpz[unz], x->x.mpz[jnew]));
                unz++;
            }
            if (loc >= k)
            {
                L->i[lnz] = jnew;
                SLIP_CHECK(SLIP_mpz_sizeinbase(&size, x->x.mpz[jnew], 2));
                SLIP_CHECK(SLIP_mpz_init2(L->x.mpz[lnz], size + 2));
                SLIP_CHECK(SLIP_mpz_set(L->x.mpz[lnz], x->x.mpz[jnew]));
                lnz++;
            }
        }
    }

    L->p[n] = lnz;
    U->p[n] = unz;

    UPDATE_FREE_WORK;

    slip_sparse_collapse(L);
    slip_sparse_collapse(U);

    // Final permutation pass: rewrite row indices in pinv-permuted space.
    for (int64_t m = 0; m < lnz; m++) L->i[m] = pinv[L->i[m]];
    for (int64_t m = 0; m < unz; m++) U->i[m] = pinv[U->i[m]];

    // Check singularity (rhos[n-1] == 0)
    int sgn;
    SLIP_CHECK(SLIP_mpz_sgn(&sgn, rhos->x.mpz[n - 1]));
    if (sgn == 0)
    {
        SLIP_matrix_free(&L, NULL);
        SLIP_matrix_free(&U, NULL);
        return SLIP_SINGULAR;
    }

    // Swap in new factors
    SLIP_matrix_free(L_handle, NULL);
    SLIP_matrix_free(U_handle, NULL);
    *L_handle = L;
    *U_handle = U;
    return SLIP_OK;
}
