//------------------------------------------------------------------------------
// SLIP_LU/Demo/sdipps.c: sparse DIP push-and-swap update for SLIP LU
//------------------------------------------------------------------------------
// Literal port of sdipps.py to C using SLIP's mpz_t wrappers. See sdipps.h for
// the public interface and sdipps.py for the algorithmic answer sheet.
//
// Layout at runtime:
//   L         : CSC MPZ, columns of old L, rows in pinv-permuted space
//   Ut        : CSC MPZ of transpose(U); Ut column i == row i of old U
//   rhos_arr  : dense mpz_t[n+1], rhos_arr[0]=1, rhos_arr[i]=rhos[i-1] (paper)
//   rhosp     : dense mpz_t[n+1], rhosp[0]=1, rhosp[1..K] = rhos_arr[1..K]
//   muL, muU  : dense mpz_t[n], seed vectors
//   hL, hU    : dense int64_t[n], history vectors
//   fP, fQ    : dense int64_t[n], row/col permutations tracked by SDIPPS
//   Ftmp      : growable COO (row, col, mpz_t) triplet list
//
// Diagonal push is preferred; column push is the fallback. This inverts the
// current sdipps.py preference (which is column-first as a temporary choice).

#include "sdipps.h"
#include "slip_internal.h"
#include <gmp.h>
#include <string.h>
#include <time.h>

// Coarse per-section timers (visible via SDIPPS_TIMING env).
static double g_t_back = 0.0, g_t_rwsop = 0.0, g_t_scatter = 0.0,
              g_t_output = 0.0, g_t_other = 0.0;
static inline double sec_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

//------------------------------------------------------------------------------
// small helpers
//------------------------------------------------------------------------------

// growable COO triplet list of mpz_t values
typedef struct {
    int64_t *row;
    int64_t *col;
    mpz_t   *val;   // each mpz_t initialized on insert
    int64_t  nz;
    int64_t  nzmax;
} sdipps_coo;

static SLIP_info coo_init(sdipps_coo *T, int64_t initial_cap)
{
    SLIP_info info;   (void) info;
    T->row = (int64_t *) SLIP_malloc(initial_cap * sizeof(int64_t));
    T->col = (int64_t *) SLIP_malloc(initial_cap * sizeof(int64_t));
    T->val = (mpz_t   *) SLIP_malloc(initial_cap * sizeof(mpz_t));
    if (!T->row || !T->col || !T->val)
    {
        SLIP_FREE(T->row); SLIP_FREE(T->col); SLIP_FREE(T->val);
        return SLIP_OUT_OF_MEMORY;
    }
    // Pre-init all mpz slots so coo_append can just swap/set into them
    // without per-call mpz_init.
    for (int64_t k = 0; k < initial_cap; k++) mpz_init(T->val[k]);
    T->nz = 0;
    T->nzmax = initial_cap;
    return SLIP_OK;
}

static SLIP_info coo_grow(sdipps_coo *T)
{
    int64_t newcap = T->nzmax * 2;
    int64_t oldcap = T->nzmax;
    bool ok;
    int64_t *nr = (int64_t *) SLIP_realloc(newcap, oldcap,
        sizeof(int64_t), T->row, &ok);
    if (!ok) return SLIP_OUT_OF_MEMORY;
    T->row = nr;
    int64_t *nc = (int64_t *) SLIP_realloc(newcap, oldcap,
        sizeof(int64_t), T->col, &ok);
    if (!ok) return SLIP_OUT_OF_MEMORY;
    T->col = nc;
    mpz_t *nv = (mpz_t *) SLIP_realloc(newcap, oldcap,
        sizeof(mpz_t), T->val, &ok);
    if (!ok) return SLIP_OUT_OF_MEMORY;
    T->val = nv;
    // Init only the newly added mpz slots
    for (int64_t k = oldcap; k < newcap; k++) mpz_init(T->val[k]);
    T->nzmax = newcap;
    return SLIP_OK;
}

// Append (r, c, v) to T with a value copy (raw mpz_set, no wrapper).
static inline SLIP_info coo_append_mpz(sdipps_coo *T, int64_t r, int64_t c,
    const mpz_t v)
{
    SLIP_info info;
    if (T->nz == T->nzmax) SLIP_CHECK(coo_grow(T));
    mpz_set(T->val[T->nz], v);
    T->row[T->nz] = r;
    T->col[T->nz] = c;
    T->nz++;
    return SLIP_OK;
}

// Append by O(1) ownership transfer via mpz_swap.
static inline SLIP_info coo_append_swap(sdipps_coo *T, int64_t r, int64_t c,
    mpz_t v)
{
    SLIP_info info;
    if (T->nz == T->nzmax) SLIP_CHECK(coo_grow(T));
    mpz_swap(T->val[T->nz], v);
    T->row[T->nz] = r;
    T->col[T->nz] = c;
    T->nz++;
    return SLIP_OK;
}

static void coo_free(sdipps_coo *T)
{
    if (!T) return;
    // Clear ALL pre-inited slots, not just 0..nz-1
    for (int64_t k = 0; k < T->nzmax; k++) mpz_clear(T->val[k]);
    SLIP_FREE(T->row); SLIP_FREE(T->col); SLIP_FREE(T->val);
    T->nz = 0; T->nzmax = 0;
}

// dense mpz_t[n] alloc/free
static SLIP_info dense_mpz_alloc(mpz_t **out, int64_t n)
{
    mpz_t *a = (mpz_t *) SLIP_calloc(n, sizeof(mpz_t));
    if (!a) return SLIP_OUT_OF_MEMORY;
    for (int64_t i = 0; i < n; i++)
    {
        SLIP_info info = SLIP_mpz_init(a[i]);
        if (info != SLIP_OK)
        {
            for (int64_t j = 0; j < i; j++) mpz_clear(a[j]);
            SLIP_FREE(a);
            return info;
        }
    }
    *out = a;
    return SLIP_OK;
}

static void dense_mpz_free(mpz_t *a, int64_t n)
{
    if (!a) return;
    for (int64_t i = 0; i < n; i++) mpz_clear(a[i]);
    SLIP_FREE(a);
}

//------------------------------------------------------------------------------
// Build Ut = CSC of transpose(U) so that Ut column i holds row i of U.
//------------------------------------------------------------------------------
static SLIP_info build_Ut(SLIP_matrix **Ut_handle, const SLIP_matrix *U,
    const SLIP_options *option)
{
    SLIP_info info;
    int64_t n = U->n;
    int64_t nnz = U->p[n];
    SLIP_matrix *Ut = NULL;
    SLIP_CHECK(SLIP_matrix_allocate(&Ut, SLIP_CSC, SLIP_MPZ, n, n, nnz,
        false, false, option));

    // count nnz per row of U -> column of Ut
    for (int64_t i = 0; i <= n; i++) Ut->p[i] = 0;
    for (int64_t k = 0; k < nnz; k++) Ut->p[U->i[k] + 1]++;
    for (int64_t i = 0; i < n; i++) Ut->p[i+1] += Ut->p[i];

    int64_t *next = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    if (!next) { SLIP_matrix_free(&Ut, option); return SLIP_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < n; i++) next[i] = Ut->p[i];

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            int64_t i = U->i[p];          // U row
            int64_t dst = next[i]++;
            Ut->i[dst] = j;               // was U col, now Ut row
            size_t sz = mpz_sizeinbase(U->x.mpz[p], 2);
            SLIP_CHECK(SLIP_mpz_init2(Ut->x.mpz[dst], sz + 2));
            mpz_set(Ut->x.mpz[dst], U->x.mpz[p]);
        }
    }
    SLIP_FREE(next);
    *Ut_handle = Ut;
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// Scatter one CSC column into a dense mpz_t[n] scratch, zeroing untouched rows.
// Returns via touched[] which rows were written (caller clears entries after).
//------------------------------------------------------------------------------
static void scatter_col(mpz_t *dense, int8_t *touched,
    const SLIP_matrix *M, int64_t j)
{
    for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
    {
        int64_t r = M->i[p];
        mpz_set(dense[r], M->x.mpz[p]);
        touched[r] = 1;
    }
}

static void clear_scatter(mpz_t *dense, int8_t *touched,
    const SLIP_matrix *M, int64_t j)
{
    for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
    {
        int64_t r = M->i[p];
        mpz_set_ui(dense[r], 0);
        touched[r] = 0;
    }
}

//------------------------------------------------------------------------------
// sLTDpush: diagonal push, "L" or "U" orientation (ind == 'L' or 'U').
//
// Ports sdipps.py sLTDpush.  In the "L" orientation:
//   - source = L (dense-scattered column k+1)
//   - writes into Ftmp at (fP[i], fQ[k])
// In the "U" orientation, called with:
//   - source = Ut column k+1 scatter (= row k+1 of U)
//   - writes into Ftmp at (fQ[k], fP[i])       <-- matches Python literally;
//     equivalent to (fP[k], fQ[i]) only when fP == fQ (all-diagonal chain).
// Uses the seed/history pair (mu, h) matching the source (muL for L, muU for U)
// and the "cross" seed value passed as sL argument (== python's sL for L call,
// == python's sU for U call — Python swaps the sL/sU args on the U call).
//------------------------------------------------------------------------------
static SLIP_info sLTDpush(char ind, int64_t k, int64_t n,
    const mpz_t *src_col,       // dense scatter of L[:][k+1] or Ut[:][k+1]
    const mpz_t *rhos,          // OLD rhos (with sign flips)
    const mpz_t *rhosp,         // NEW rhos
    const int64_t *fP, const int64_t *fQ,
    sdipps_coo *Ftmp,
    const mpz_t sL_arg,         // sL for L-call, sU for U-call (Python order)
    const mpz_t sU_arg,         // sU for L-call, sL for U-call
    mpz_t *mu, int64_t *h,
    mpz_t tmp)
{
    SLIP_info info;
    int sU_sgn = mpz_sgn(sU_arg);

    // -- backtracking (Ftmp writes for i in k+2..n-1) --
    if (sU_sgn == 0)
    {
        for (int64_t i = k+2; i < n; i++)
        {
            if (mpz_sgn(src_col[i]) == 0) continue;  // tmp would be 0
            // Ftmp[dst] = (rhosp[k] * src_col[i]) / rhos[k+1]
            mpz_mul(tmp, rhosp[k], src_col[i]);
            mpz_divexact(tmp, tmp, rhos[k+1]);
            if (mpz_sgn(tmp) == 0) continue;
            int64_t r = (ind == 'L') ? fP[i] : fQ[k];
            int64_t c = (ind == 'L') ? fQ[k] : fP[i];
            SLIP_CHECK(coo_append_swap(Ftmp, r, c, tmp));
        }
    }
    else
    {
        for (int64_t i = k+2; i < n; i++)
        {
            int mu_sgn = mpz_sgn(mu[i]);
            int src_sgn = mpz_sgn(src_col[i]);
            if (mu_sgn == 0 && src_sgn == 0) continue;  // both zero → tmp=0
            if (h[i] < k - 1 && mu_sgn != 0)
            {
                mpz_mul(mu[i], rhos[k+1], mu[i]);
                mpz_divexact(mu[i], mu[i], rhos[h[i]+2]);
                h[i] = k - 1;
            }
            // Ftmp[dst] = (rhosp[k]*src_col[i] + sU_arg*mu[i]) / rhos[k+1]
            if (src_sgn != 0)
            {
                mpz_mul(tmp, rhosp[k], src_col[i]);
            }
            else
            {
                mpz_set_ui(tmp, 0);
            }
            if (mu_sgn != 0)
                mpz_addmul(tmp, sU_arg, mu[i]);
            if (mpz_sgn(tmp) == 0) continue;
            mpz_divexact(tmp, tmp, rhos[k+1]);
            if (mpz_sgn(tmp) == 0) continue;
            int64_t r = (ind == 'L') ? fP[i] : fQ[k];
            int64_t c = (ind == 'L') ? fQ[k] : fP[i];
            SLIP_CHECK(coo_append_swap(Ftmp, r, c, tmp));
        }
    }

    // -- RwSOP (seed vector update) --
    if (k < n - 2)
    {
        int sL_sgn = mpz_sgn(sL_arg);
        if (sL_sgn == 0)
        {
            // just bring mu[k+2] up to history k
            if (mpz_sgn(mu[k+2]) != 0)
            {
                mpz_mul(mu[k+2], rhos[k+2], mu[k+2]);
                mpz_divexact(mu[k+2], mu[k+2], rhos[h[k+2]+2]);
            }
            h[k+2] = k;
        }
        else
        {
            int mu_sgn = mpz_sgn(mu[k+2]);
            if (h[k+2] != k - 1 && mu_sgn != 0)
            {
                mpz_mul(mu[k+2], rhos[k+1], mu[k+2]);
                mpz_divexact(mu[k+2], mu[k+2], rhos[h[k+2]+2]);
                h[k+2] = k - 1;
            }
            // mu[k+2] = (rhos[k+2]*mu[k+2] - sL*src_col[k+2]) / rhos[k+1]
            if (mu_sgn != 0)
                mpz_mul(mu[k+2], rhos[k+2], mu[k+2]);
            if (mpz_sgn(src_col[k+2]) != 0)
                mpz_submul(mu[k+2], sL_arg, src_col[k+2]);
            if (mpz_sgn(mu[k+2]) != 0)
                mpz_divexact(mu[k+2], mu[k+2], rhos[k+1]);
            h[k+2] = k;

            for (int64_t i = k+3; i < n; i++)
            {
                int mi = mpz_sgn(mu[i]);
                int si = mpz_sgn(src_col[i]);
                if (mi == 0 && si == 0) { h[i] = k; continue; }
                if (h[i] != k - 1 && mi != 0)
                {
                    mpz_mul(mu[i], rhos[k+1], mu[i]);
                    mpz_divexact(mu[i], mu[i], rhos[h[i]+2]);
                }
                if (mi != 0)
                    mpz_mul(mu[i], rhos[k+2], mu[i]);
                if (si != 0)
                    mpz_submul(mu[i], sL_arg, src_col[i]);
                if (mpz_sgn(mu[i]) != 0)
                    mpz_divexact(mu[i], mu[i], rhos[k+1]);
                h[i] = k;
            }
        }
    }
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// sLTCpush: column-push L helper.
//   Backtracking writes into Ftmp column fQ[k]; RwSOP re-initializes muL to
//   -L[:][k+1] with h = k.
//------------------------------------------------------------------------------
static SLIP_info sLTCpush(int64_t k, int64_t n,
    const mpz_t *L_col_kp1,     // dense scatter of L[:][k+1]
    const mpz_t *rhos, const mpz_t *rhosp,
    const int64_t *fP, const int64_t *fQ,
    sdipps_coo *Ftmp,
    const mpz_t sL, const mpz_t sU,
    mpz_t *muL, int64_t *hL,
    mpz_t tmp)
{
    SLIP_info info;
    int sU_sgn = mpz_sgn(sU);

    for (int64_t i = k+2; i < n; i++)
    {
        int src_sgn = mpz_sgn(L_col_kp1[i]);
        int mu_sgn = sU_sgn != 0 ? mpz_sgn(muL[i]) : 0;
        if (src_sgn == 0 && mu_sgn == 0) continue;
        if (sU_sgn != 0)
        {
            if (hL[i] != k - 1 && mu_sgn != 0)
            {
                mpz_mul(muL[i], rhos[k+1], muL[i]);
                mpz_divexact(muL[i], muL[i], rhos[hL[i]+2]);
                hL[i] = k - 1;
            }
            if (src_sgn != 0)
            {
                mpz_mul(tmp, rhosp[k], L_col_kp1[i]);
            }
            else
            {
                mpz_set_ui(tmp, 0);
            }
            if (mu_sgn != 0)
                mpz_addmul(tmp, sU, muL[i]);
        }
        else
        {
            mpz_mul(tmp, rhosp[k], L_col_kp1[i]);
        }
        if (mpz_sgn(tmp) == 0) continue;
        mpz_divexact(tmp, tmp, rhos[k+1]);
        if (mpz_sgn(tmp) == 0) continue;
        SLIP_CHECK(coo_append_swap(Ftmp, fP[i], fQ[k], tmp));
    }

    // RwSOP: reinitialize muL to -L[:][k+1] with hL = k
    if (k < n - 2)
    {
        for (int64_t i = 0; i < n; i++)
        {
            if (mpz_sgn(L_col_kp1[i]) != 0)
            {
                mpz_set(muL[i], L_col_kp1[i]);
                mpz_neg(muL[i], muL[i]);
            }
            else
            {
                mpz_set_ui(muL[i], 0);
            }
            hL[i] = k;
        }
    }
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// sUTCpush: column-push U helper.
//   Backtracking copies muU into Ftmp row fP[k].
//   RwSOP updates muU via the sign-flipped formula.
//------------------------------------------------------------------------------
static SLIP_info sUTCpush(int64_t k, int64_t n,
    const mpz_t *Ut_col_kp1,    // dense scatter of Ut[:][k+1] = row k+1 of U
    const mpz_t *rhos, const mpz_t *rhosp,
    const int64_t *fP, const int64_t *fQ,
    sdipps_coo *Ftmp,
    const mpz_t sL, const mpz_t sU,
    mpz_t *muU, int64_t *hU,
    mpz_t tmp)
{
    SLIP_info info;
    (void) rhosp; (void) sL;    // sL/rhosp unused in sUTCpush per sdipps.py
    for (int64_t i = k+2; i < n; i++)
    {
        int mu_sgn = mpz_sgn(muU[i]);
        if (mu_sgn == 0) continue;
        if (hU[i] != k - 1)
        {
            mpz_mul(muU[i], rhos[k+1], muU[i]);
            mpz_divexact(muU[i], muU[i], rhos[hU[i]+2]);
            hU[i] = k - 1;
        }
        if (mpz_sgn(muU[i]) != 0)
            SLIP_CHECK(coo_append_mpz(Ftmp, fP[k], fQ[i], muU[i]));
    }

    if (k < n - 2)
    {
        int sU_sgn = mpz_sgn(sU);
        if (sU_sgn == 0)
        {
            if (mpz_sgn(muU[k+2]) != 0)
            {
                mpz_mul(muU[k+2], rhos[k+2], muU[k+2]);
                mpz_divexact(muU[k+2], muU[k+2], rhos[hU[k+2]+2]);
            }
            hU[k+2] = k;
        }
        else
        {
            int mu_sgn = mpz_sgn(muU[k+2]);
            if (hU[k+2] != k - 1 && mu_sgn != 0)
            {
                mpz_mul(muU[k+2], rhos[k+1], muU[k+2]);
                mpz_divexact(muU[k+2], muU[k+2], rhos[hU[k+2]+2]);
                hU[k+2] = k - 1;
            }
            // muU[k+2] = -(rhos[k+2]*muU[k+2] - sU*Ut[k+2][k+1]) / rhos[k+1]
            if (mu_sgn != 0)
                mpz_mul(muU[k+2], rhos[k+2], muU[k+2]);
            if (mpz_sgn(Ut_col_kp1[k+2]) != 0)
                mpz_submul(muU[k+2], sU, Ut_col_kp1[k+2]);
            if (mpz_sgn(muU[k+2]) != 0)
            {
                mpz_divexact(muU[k+2], muU[k+2], rhos[k+1]);
                mpz_neg(muU[k+2], muU[k+2]);
            }
            hU[k+2] = k;

            for (int64_t i = k+3; i < n; i++)
            {
                int mi = mpz_sgn(muU[i]);
                int si = mpz_sgn(Ut_col_kp1[i]);
                if (mi == 0 && si == 0) { hU[i] = k; continue; }
                if (hU[i] != k - 1 && mi != 0)
                {
                    mpz_mul(muU[i], rhos[k+1], muU[i]);
                    mpz_divexact(muU[i], muU[i], rhos[hU[i]+2]);
                }
                if (mi != 0)
                    mpz_mul(muU[i], rhos[k+2], muU[i]);
                if (si != 0)
                    mpz_submul(muU[i], sU, Ut_col_kp1[i]);
                if (mpz_sgn(muU[i]) != 0)
                {
                    mpz_divexact(muU[i], muU[i], rhos[k+1]);
                    mpz_neg(muU[i], muU[i]);
                }
                hU[i] = k;
            }
        }
    }
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// Convert Ftmp COO -> CSC L_new and CSC U_new by applying inverse of fP, fQ
// so that F[i][j] = Ftmp[fP[i]][fQ[j]].  Splits into L (i>=j) and U (i<=j),
// with diagonal duplicated in both.  Row indices of the output are then
// re-permuted by pinv (to match SLIP_LU_factorize's output convention).
//------------------------------------------------------------------------------
static SLIP_info coo_to_LU_csc(SLIP_matrix **L_out, SLIP_matrix **U_out,
    sdipps_coo *Ftmp, const int64_t *fP, const int64_t *fQ,
    const int64_t *pinv, int64_t n, const SLIP_options *option)
{
    SLIP_info info;
    // build inverse permutations
    int64_t *fPinv = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    int64_t *fQinv = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    if (!fPinv || !fQinv) { SLIP_FREE(fPinv); SLIP_FREE(fQinv);
        return SLIP_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < n; i++) { fPinv[fP[i]] = i; fQinv[fQ[i]] = i; }

    // count nnz per output column, separated into L and U
    int64_t *Lp = (int64_t *) SLIP_calloc(n + 1, sizeof(int64_t));
    int64_t *Up = (int64_t *) SLIP_calloc(n + 1, sizeof(int64_t));
    if (!Lp || !Up) { SLIP_FREE(fPinv); SLIP_FREE(fQinv);
        SLIP_FREE(Lp); SLIP_FREE(Up); return SLIP_OUT_OF_MEMORY; }

    for (int64_t k = 0; k < Ftmp->nz; k++)
    {
        int64_t i_out = fPinv[Ftmp->row[k]];
        int64_t j_out = fQinv[Ftmp->col[k]];
        if (i_out >= j_out) Lp[j_out + 1]++;
        if (i_out <= j_out) Up[j_out + 1]++;
    }
    for (int64_t j = 0; j < n; j++) { Lp[j+1] += Lp[j]; Up[j+1] += Up[j]; }
    int64_t Lnz = Lp[n], Unz = Up[n];

    SLIP_matrix *L = NULL, *U = NULL;
    info = SLIP_matrix_allocate(&L, SLIP_CSC, SLIP_MPZ, n, n,
        (Lnz > 0 ? Lnz : 1), false, false, option);
    if (info != SLIP_OK) { SLIP_FREE(fPinv); SLIP_FREE(fQinv);
        SLIP_FREE(Lp); SLIP_FREE(Up); return info; }
    info = SLIP_matrix_allocate(&U, SLIP_CSC, SLIP_MPZ, n, n,
        (Unz > 0 ? Unz : 1), false, false, option);
    if (info != SLIP_OK) { SLIP_matrix_free(&L, option);
        SLIP_FREE(fPinv); SLIP_FREE(fQinv);
        SLIP_FREE(Lp); SLIP_FREE(Up); return info; }

    for (int64_t j = 0; j <= n; j++) L->p[j] = Lp[j];
    for (int64_t j = 0; j <= n; j++) U->p[j] = Up[j];

    int64_t *Lnext = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    int64_t *Unext = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    if (!Lnext || !Unext) { SLIP_FREE(Lnext); SLIP_FREE(Unext);
        SLIP_matrix_free(&L, option); SLIP_matrix_free(&U, option);
        SLIP_FREE(fPinv); SLIP_FREE(fQinv);
        SLIP_FREE(Lp); SLIP_FREE(Up); return SLIP_OUT_OF_MEMORY; }
    for (int64_t j = 0; j < n; j++) { Lnext[j] = L->p[j]; Unext[j] = U->p[j]; }

    (void) pinv;   // Ftmp row space is already in the composed (input-pinv ∘ fP)
                   // permuted space, so we just undo fP; no additional pinv step.
    // Ownership transfer: mpz_swap the bignum out of Ftmp->val[k] into the
    // destination slot (O(1) — just swaps limb pointers). For diagonal entries
    // (i_out == j_out) that go into both L and U, swap into L and copy into U.
    for (int64_t k = 0; k < Ftmp->nz; k++)
    {
        int64_t i_out = fPinv[Ftmp->row[k]];
        int64_t j_out = fQinv[Ftmp->col[k]];
        int64_t i_stored = i_out;
        if (i_out > j_out)
        {
            // L only: swap ownership
            int64_t p = Lnext[j_out]++;
            L->i[p] = i_stored;
            mpz_init(L->x.mpz[p]);
            mpz_swap(L->x.mpz[p], Ftmp->val[k]);
        }
        else if (i_out < j_out)
        {
            // U only
            int64_t p = Unext[j_out]++;
            U->i[p] = i_stored;
            mpz_init(U->x.mpz[p]);
            mpz_swap(U->x.mpz[p], Ftmp->val[k]);
        }
        else
        {
            // Diagonal: swap into L, copy into U (one bignum copy per diagonal)
            int64_t pL = Lnext[j_out]++;
            int64_t pU = Unext[j_out]++;
            L->i[pL] = i_stored;
            U->i[pU] = i_stored;
            mpz_init(L->x.mpz[pL]);
            mpz_swap(L->x.mpz[pL], Ftmp->val[k]);
            mpz_init(U->x.mpz[pU]);
            mpz_set(U->x.mpz[pU], L->x.mpz[pL]);
        }
    }

    SLIP_FREE(Lnext); SLIP_FREE(Unext);
    SLIP_FREE(Lp); SLIP_FREE(Up);
    SLIP_FREE(fPinv); SLIP_FREE(fQinv);
    *L_out = L; *U_out = U;
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// SDIPPS: main entry
//------------------------------------------------------------------------------
SLIP_info SDIPPS
(
    SLIP_matrix **L_handle,
    SLIP_matrix **U_handle,
    SLIP_matrix  *rhos,
    int64_t      *pinv,
    int64_t      *q,
    const SLIP_matrix *A_new,
    int64_t       K,
    const SLIP_options *option
)
{
    if (!slip_initialized()) return SLIP_PANIC;
    if (!L_handle || !U_handle || !*L_handle || !*U_handle
        || !rhos || !pinv || !q)
        return SLIP_INCORRECT_INPUT;

    SLIP_matrix *L = *L_handle;
    SLIP_matrix *U = *U_handle;
    SLIP_REQUIRE(L, SLIP_CSC, SLIP_MPZ);
    SLIP_REQUIRE(U, SLIP_CSC, SLIP_MPZ);
    (void) A_new;   // unused inside SDIPPS core; caller may still supply

    int64_t n = L->n;
    if (U->n != n || rhos->m != n || K < 0 || K >= n)
        return SLIP_INCORRECT_INPUT;

    SLIP_info info;
    SLIP_matrix *Ut = NULL;
    mpz_t *rhos_arr = NULL, *rhosp = NULL;
    mpz_t *muL = NULL, *muU = NULL;
    int64_t *hL = NULL, *hU = NULL;
    int64_t *fP = NULL, *fQ = NULL;
    int64_t *row_perm = NULL;
    mpz_t *Lscat = NULL, *Uscat = NULL;
    int8_t *Ltouched = NULL, *Utouched = NULL;
    sdipps_coo Ftmp = { .row=NULL, .col=NULL, .val=NULL, .nz=0, .nzmax=0 };
    mpz_t tmp, sL, sU, diag_cond;
    int tmp_init = 0, scratch_init = 0;
    SLIP_matrix *L_new = NULL, *U_new = NULL;

    #define CLEANUP                                                       \
    {                                                                     \
        SLIP_matrix_free(&Ut, option);                                    \
        dense_mpz_free(rhos_arr, n + 1);                                  \
        dense_mpz_free(rhosp,    n + 1);                                  \
        dense_mpz_free(muL, n); dense_mpz_free(muU, n);                   \
        SLIP_FREE(hL); SLIP_FREE(hU);                                     \
        SLIP_FREE(fP); SLIP_FREE(fQ);                                     \
        dense_mpz_free(Lscat, n); dense_mpz_free(Uscat, n);               \
        SLIP_FREE(Ltouched); SLIP_FREE(Utouched);                         \
        SLIP_FREE(row_perm);                                              \
        coo_free(&Ftmp);                                                  \
        if (tmp_init) mpz_clear(tmp);                                     \
        if (scratch_init) { /*scratch stays live*/ } \
    }

    // build Ut = CSC of transpose(U)
    info = build_Ut(&Ut, U, option);
    if (info != SLIP_OK) { CLEANUP; return info; }

    // rhos_arr[0]=1, rhos_arr[i+1] = rhos[i]  (Python indexing convention)
    info = dense_mpz_alloc(&rhos_arr, n + 1); if (info) { CLEANUP; return info; }
    info = dense_mpz_alloc(&rhosp,    n + 1); if (info) { CLEANUP; return info; }
    mpz_set_ui(rhos_arr[0], 1);
    mpz_set_ui(rhosp[0],    1);
    for (int64_t i = 0; i < n; i++)
        mpz_set(rhos_arr[i+1], rhos->x.mpz[i]);
    for (int64_t i = 0; i < K; i++)
        mpz_set(rhosp[i+1], rhos_arr[i+1]);

    // seed / history
    info = dense_mpz_alloc(&muL, n); if (info) { CLEANUP; return info; }
    info = dense_mpz_alloc(&muU, n); if (info) { CLEANUP; return info; }
    hL = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    hU = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    fP = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    fQ = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    if (!hL || !hU || !fP || !fQ) { CLEANUP; return SLIP_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < n; i++) { hL[i] = K - 1; hU[i] = K - 1;
        fP[i] = i; fQ[i] = i; }

    // row_perm: inverse of pinv (position -> original row), needed to update
    // pinv in place on each diagonal push (which swaps two factored positions).
    row_perm = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
    if (!row_perm) { CLEANUP; return SLIP_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < n; i++) row_perm[pinv[i]] = i;

    // muL = column K of L; muU = column K of Ut (= row K of U)
    for (int64_t p = L->p[K]; p < L->p[K+1]; p++)
        mpz_set(muL[L->i[p]], L->x.mpz[p]);
    for (int64_t p = Ut->p[K]; p < Ut->p[K+1]; p++)
        mpz_set(muU[Ut->i[p]], Ut->x.mpz[p]);

    // dense scratch for scattering L / Ut columns each iteration
    info = dense_mpz_alloc(&Lscat, n); if (info) { CLEANUP; return info; }
    info = dense_mpz_alloc(&Uscat, n); if (info) { CLEANUP; return info; }
    Ltouched = (int8_t *) SLIP_calloc(n, sizeof(int8_t));
    Utouched = (int8_t *) SLIP_calloc(n, sizeof(int8_t));
    if (!Ltouched || !Utouched) { CLEANUP; return SLIP_OUT_OF_MEMORY; }

    // Ftmp: seed capacity, will grow
    info = coo_init(&Ftmp, 8 * n);
    if (info != SLIP_OK) { CLEANUP; return info; }

    // Pre-load prefix: for k in 0..K-1, copy L col k and U row k into Ftmp
    for (int64_t kk = 0; kk < K; kk++)
    {
        for (int64_t p = L->p[kk]; p < L->p[kk+1]; p++)
        {
            int64_t i = L->i[p];
            if (i >= kk) SLIP_CHECK(coo_append_mpz(&Ftmp, i, kk, L->x.mpz[p]));
        }
        for (int64_t p = Ut->p[kk]; p < Ut->p[kk+1]; p++)
        {
            int64_t j = Ut->i[p];
            if (j > kk) SLIP_CHECK(coo_append_mpz(&Ftmp, kk, j, Ut->x.mpz[p]));
        }
    }

    mpz_init(tmp); tmp_init = 1;

    // Hoisted per-iteration scratch (init once, reuse via mpz_set).
    mpz_init(sL);
    mpz_init(sU);
    mpz_init(diag_cond);
    scratch_init = 1;

    // Main push loop
    for (int64_t k = K; k < n - 1; k++)
    {
        // sL = muL[k+1], sU = muU[k+1]  (bring-up not needed: invariant h=k-1)
        mpz_set(sL, muL[k+1]);
        mpz_set(sU, muU[k+1]);
        int sL_sgn = mpz_sgn(sL);
        int sU_sgn = mpz_sgn(sU);

        // diagonal condition: rhosp[k]*rhos[k+2] + sL*sU != 0
        mpz_mul(diag_cond, rhosp[k], rhos_arr[k+2]);
        mpz_addmul(diag_cond, sL, sU);
        int diag_sgn = mpz_sgn(diag_cond);

        // scatter L col k+1 and Ut col k+1 into dense
        double _t0 = sec_now();
        scatter_col(Lscat, Ltouched, L, k+1);
        scatter_col(Uscat, Utouched, Ut, k+1);
        g_t_scatter += sec_now() - _t0;

        if (diag_sgn != 0)
        {
            // ------ Diagonal push ------
            // fP swap k <-> k+1, fQ swap k <-> k+1
            { int64_t t = fP[k]; fP[k] = fP[k+1]; fP[k+1] = t; }
            { int64_t t = fQ[k]; fQ[k] = fQ[k+1]; fQ[k+1] = t; }
            { int64_t t = q[k];  q[k]  = q[k+1];  q[k+1]  = t; }
            // pinv update: the row currently at factored position k moves to
            // position k+1 and vice versa.
            {
                int64_t r0 = row_perm[k], r1 = row_perm[k+1];
                row_perm[k] = r1; row_perm[k+1] = r0;
                pinv[r0] = k+1;   pinv[r1] = k;
            }

            // rhosp[k+1] = (rhosp[k]*rhos[k+2] + sL*sU) / rhos[k+1]
            mpz_divexact(rhosp[k+1], diag_cond, rhos_arr[k+1]);

            // three corner writes in Ftmp
            SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k], fQ[k], rhosp[k+1]));
            if (sL_sgn != 0)
                SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k], fQ[k+1], sL));
            if (sU_sgn != 0)
                SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k+1], fQ[k], sU));

            // L push: sL_arg=sL, sU_arg=sU, source=Lscat
            double _tp = sec_now();
            info = sLTDpush('L', k, n, (const mpz_t *) Lscat,
                (const mpz_t *) rhos_arr, (const mpz_t *) rhosp,
                fP, fQ, &Ftmp, sL, sU, muL, hL, tmp);
            if (info) { /*scratch stays live*/
                CLEANUP; return info; }
            // U push: Python swaps sL/sU args on the U call. Source=Uscat.
            info = sLTDpush('U', k, n, (const mpz_t *) Uscat,
                (const mpz_t *) rhos_arr, (const mpz_t *) rhosp,
                fP, fQ, &Ftmp, sU, sL, muU, hU, tmp);
            g_t_rwsop += sec_now() - _tp;
            if (info) { /*scratch stays live*/
                CLEANUP; return info; }
            // (scratch mpzs sL/sU/diag_cond stay live across iterations)
        }
        else if (sU_sgn != 0)
        {
            // ------ Column push ------
            { int64_t t = fQ[k]; fQ[k] = fQ[k+1]; fQ[k+1] = t; }
            { int64_t t = q[k];  q[k]  = q[k+1];  q[k+1]  = t; }

            // rhosp[k+1] = sU
            mpz_set(rhosp[k+1], sU);

            // three corner writes
            SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k], fQ[k], sU));
            SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k], fQ[k+1], rhos_arr[k+1]));
            // Ftmp[fP[k+1]][fQ[k]] = (rhosp[k]*rhos[k+2] + sU*sL)/rhos[k+1]
            mpz_divexact(tmp, diag_cond, rhos_arr[k+1]);
            SLIP_CHECK(coo_append_mpz(&Ftmp, fP[k+1], fQ[k], tmp));

            info = sLTCpush(k, n, (const mpz_t *) Lscat,
                (const mpz_t *) rhos_arr, (const mpz_t *) rhosp,
                fP, fQ, &Ftmp, sL, sU, muL, hL, tmp);
            if (info) { /*scratch stays live*/
                CLEANUP; return info; }
            info = sUTCpush(k, n, (const mpz_t *) Uscat,
                (const mpz_t *) rhos_arr, (const mpz_t *) rhosp,
                fP, fQ, &Ftmp, sL, sU, muU, hU, tmp);
            if (info) { /*scratch stays live*/
                CLEANUP; return info; }

            // sign flip: rhos[k+2..n], L tail cols, Ut tail cols  (per sdipps.py)
            mpz_neg(rhos_arr[k+2], rhos_arr[k+2]);
            for (int64_t j = k+2; j < n; j++)
            {
                if (j + 1 <= n) mpz_neg(rhos_arr[j+1], rhos_arr[j+1]);
                for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
                    if (L->i[p] >= j) mpz_neg(L->x.mpz[p], L->x.mpz[p]);
                for (int64_t p = Ut->p[j]; p < Ut->p[j+1]; p++)
                    if (Ut->i[p] >= j) mpz_neg(Ut->x.mpz[p], Ut->x.mpz[p]);
            }
        }
        else
        {
            /*scratch stays live*/
            CLEANUP;
            return SLIP_SINGULAR;   // no push available
        }

        // clear scatters for next iter
        clear_scatter(Lscat, Ltouched, L, k+1);
        clear_scatter(Uscat, Utouched, Ut, k+1);

        /*scratch stays live*/
    }

    // Post-loop write: Ftmp[fP[n-1]][fQ[n-1]] = rhos[n] = rhos_arr[n]
    SLIP_CHECK(coo_append_mpz(&Ftmp, fP[n-1], fQ[n-1], rhos_arr[n]));
    // Also set rhosp[n] = rhos_arr[n] so downstream sees the final pivot
    mpz_set(rhosp[n], rhos_arr[n]);

    // Rewrite output rhos vector from rhosp[1..n]
    for (int64_t i = 0; i < n; i++)
        mpz_set(rhos->x.mpz[i], rhosp[i+1]);

    // Build new L, U CSC from Ftmp (applies inverse fP/fQ, then pinv on rows)
    double _to = sec_now();
    info = coo_to_LU_csc(&L_new, &U_new, &Ftmp, fP, fQ, pinv, n, option);
    if (info != SLIP_OK) { CLEANUP; return info; }
    g_t_output += sec_now() - _to;

    if (getenv("SDIPPS_TIMING"))
        fprintf(stderr,
            "[timing] scatter=%.3fs  push(L+U helpers)=%.3fs  output=%.3fs\n",
            g_t_scatter, g_t_rwsop, g_t_output);

    // Swap in new factors
    SLIP_matrix_free(L_handle, option);
    SLIP_matrix_free(U_handle, option);
    *L_handle = L_new;
    *U_handle = U_new;

    CLEANUP;
    #undef CLEANUP
    return SLIP_OK;
}
