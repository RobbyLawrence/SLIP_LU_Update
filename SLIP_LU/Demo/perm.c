//------------------------------------------------------------------------------
// SLIP_LU/Demo/perm.c: adjacent-column push update (APCPU) for SLIP LU
//------------------------------------------------------------------------------

#include "SLIP_LU.h"
#include "demos.h"
#include <stdio.h>
#include <string.h>

// Grids larger than this are elided (QSOpt_ex bases can be ~2000x2000).
#define PRINT_MAX_N 20

// The rational L*D^(-1)*U reconstruction is O(n^3) mpq work; it is skipped
// for matrices bigger than this, leaving the entrywise factor comparison
// with the from-scratch refactorization as the correctness test.
#define RECON_MAX_N 200

/* This program reads a sparse matrix stored in triplet format, factors it with
 * SLIP LU (PAQ = L*D^(-1)*U), and then demonstrates the Adjacent-Column Push
 * Update (APCPU): given the factors of PAQ, update them to the factors of
 * PAQ' -- PAQ with two ADJACENT columns k and k+1 swapped -- without
 * refactorizing from scratch.
 *
 *   1. Factor A "normally" (COLAMD column ordering + tolerance pivoting) to
 *      obtain the column ordering Q and the row permutation P.
 *   2. Form the matrix that SLIP actually factors, PAQ, explicitly.
 *   3. Ask the user for the swap position k (1 <= k <= n-1); swap columns
 *      k and k+1 of PAQ.
 *   4. Run the SPARSE APCPU directly on the CSC factors: the back-solve
 *      trick derives the new L column k and U row k+1 purely from stored
 *      entries (one inverted IPGE step), so the update never reads PAQ and
 *      costs O(nnz of the affected rows/columns), independent of k.
 *   5. Run the dense APCPU as well (when the dense workspace fits) and
 *      cross-check the two implementations entry by entry.
 *   6. Verify: (a) reconstruct L'*D'^(-1)*U' and compare with the swapped
 *      PAQ; (b) re-factor the swapped PAQ from scratch (identity Q, diagonal
 *      pivoting) and compare the factors entrywise -- the sparse-vs-scratch
 *      comparison runs at any size.
 *
 * Usage:
 *      perm [-Q] [-D] [matrix_file [k]]
 *      perm -B [-Q] matrix_file [k m]
 *      perm -R [-Q] [matrix_file [col]]
 *
 * With -D the demo performs the adjacent DIAGONAL push (APDPU) instead: the
 * symmetric permutation PAQ' = E*PAQ*E swapping both columns AND rows k,
 * k+1.  The factors are updated by the sparse and dense APDPU (closed-form
 * corner + stored-value swaps + four IPGE strips, no sign flips), with the
 * same cross-checks and from-scratch verification as the column push.
 *
 * With -B the demo runs the DISTANT diagonal push benchmark instead of the
 * single-swap workflow: the row/column at position k of PAQ is moved to
 * position m (a symmetric cyclic rotation of the positions in between) by a
 * chain of |m - k| adjacent diagonal pushes, and the chained update is timed
 * against a full refactorization of the permuted matrix (fresh COLAMD
 * analysis + factorization -- what a solver without the update would run).
 * With k and m on the command line a single push is benchmarked; otherwise a
 * sweep over distances d = 1, 2, 4, ... plus the two full-length pushes
 * 1 -> n and n -> 1 is run, and the results are printed as a table
 * comparing time and nnz.  Every trial is verified exactly (see bench_push).
 *
 * With -R the demo performs a COLUMN REPLACEMENT (the simplex basis
 * exchange): it reads a basis (default: cycle_Bases/basis_k0_B.txt), asks
 * which column to replace (or takes it on the command line), replaces that
 * column with a DENSE deterministic pseudorandom column, and updates the
 * factorization by the exact Forrest-Tomlin analogue -- a MIXED distant
 * push of the leaving position to the end (a diagonal push at every step,
 * except that where its pivot vanishes the step becomes a COLUMN push,
 * whose pivot -- the upper support U(j,j+1) -- is then provably nonzero:
 * one of the two is ALWAYS feasible on a nonsingular basis, so the chain
 * never dies on a zero pivot) followed by one REF forward solve for the
 * new last column of U -- timed against a full refactorization of the new
 * basis, with the same exactness verification as -B (see bench_replace).
 *
 * If no matrix file is given on the command line, the matrix is read from
 * "../ExampleMats/test_mat2.txt".  If k is given as the last argument it is
 * used as the swap position; otherwise the program prompts for it.
 *
 * With -Q the matrix file is a basis snapshot emitted by QSOpt_ex (see e.g.
 * cycle_Bases/basis_k0_B.txt and co9_Bases/): a rational matrix stored
 * either as 0-indexed triplets with integer-or-"p/q" values (the current
 * emitter, cycle_Bases/) or as a header, per-row denominators, and integer
 * triplets (the older emitter, 50v-10_LP_Bases/ and co9_Bases/); the two
 * are told apart automatically (see read_qsx_basis and the format sniffer
 * in main).  Either way the basis is converted to SLIP's CSC MPZ form by
 * multiplying every entry by one integer scalar (the LCM of the
 * denominators, recorded in A->scale) and the same workflow is run on it.  Since these bases are large (~2000x2000), the
 * dense grids are only printed for matrices up to 20-by-20, and the O(n^3)
 * rational reconstruction checks are skipped above n = 200 -- the entrywise
 * comparison with the from-scratch refactorization remains the correctness
 * test at any size.
 */

//------------------------------------------------------------------------------
// printing / swapping / comparison helpers
//------------------------------------------------------------------------------

// Forward declaration: defined below near reconstruct(); used by the
// verification blocks in bench_push and bench_replace, which sit above it.
static SLIP_info probe_LDU_eq_M_csc (bool *ok, const SLIP_matrix *L,
    const SLIP_matrix *U, const SLIP_matrix *rhos, const SLIP_matrix *M,
    uint64_t seed, const SLIP_options *option);

/* Print a dense SLIP_MPZ matrix as a 2D grid (rows down, columns across),
 * right-aligning every entry to the width of the widest entry so the columns
 * line up.  The dense matrix stores A(i,j) at A->x.mpz[i + j*A->m], which the
 * SLIP_2D macro accesses for us.
 */
static void print_dense_grid (const char *name, const SLIP_matrix *M)
{
    int64_t m = M->m, n = M->n;

    // find the width (in characters) of the widest entry
    int width = 1;
    for (int64_t k = 0; k < m * n; k++)
    {
        int len = (int) mpz_sizeinbase (M->x.mpz[k], 10)
                + (mpz_sgn (M->x.mpz[k]) < 0 ? 1 : 0);   // +1 for the '-' sign
        if (len > width) width = len;
    }

    printf ("\n%s (%"PRId64"-by-%"PRId64"):\n", name, m, n);
    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            gmp_printf ("%*Zd ", width, SLIP_2D (M, i, j, mpz));
        }
        printf ("\n");
    }
}

/* print_dense_grid, but elide matrices too large to read as a grid. */
static void print_dense_gated (const char *name, const SLIP_matrix *M)
{
    if (M->m > PRINT_MAX_N || M->n > PRINT_MAX_N)
    {
        printf ("\n%s (%"PRId64"-by-%"PRId64"): not printed (larger than "
            "%d-by-%d)\n", name, M->m, M->n, PRINT_MAX_N, PRINT_MAX_N);
        return;
    }
    print_dense_grid (name, M);
}

/* Swap two columns of a dense matrix in place (no-op if c1 == c2). */
static void swap_cols (SLIP_matrix *M, int64_t c1, int64_t c2)
{
    if (c1 == c2) return;
    for (int64_t i = 0; i < M->m; i++)
    {
        mpz_swap (SLIP_2D (M, i, c1, mpz), SLIP_2D (M, i, c2, mpz));
    }
}

/* Swap two rows of a dense matrix in place (no-op if r1 == r2). */
static void swap_rows (SLIP_matrix *M, int64_t r1, int64_t r2)
{
    if (r1 == r2) return;
    for (int64_t j = 0; j < M->n; j++)
    {
        mpz_swap (SLIP_2D (M, r1, j, mpz), SLIP_2D (M, r2, j, mpz));
    }
}

/* Entrywise equality of two dense MPZ matrices of the same shape. */
static bool equal_dense (const SLIP_matrix *X, const SLIP_matrix *Y)
{
    if (X->m != Y->m || X->n != Y->n) return false;
    for (int64_t t = 0; t < X->m * X->n; t++)
    {
        if (mpz_cmp (X->x.mpz[t], Y->x.mpz[t]) != 0) return false;
    }
    return true;
}

/* Entrywise equality of two CSC MPZ matrices as mathematical objects:
 * explicit zeros and the ordering of row indices within a column are
 * ignored.  Uses an O(m) scatter workspace.
 */
static bool csc_equal (const SLIP_matrix *X, const SLIP_matrix *Y)
{
    if (X->m != Y->m || X->n != Y->n) return false;
    int64_t *w = (int64_t *) SLIP_malloc (X->m * sizeof (int64_t));
    if (w == NULL) return false;
    for (int64_t i = 0; i < X->m; i++) w[i] = -1;

    bool eq = true;
    for (int64_t j = 0; j < X->n && eq; j++)
    {
        for (int64_t p = X->p[j]; p < X->p[j+1]; p++) w[X->i[p]] = p;
        for (int64_t p = Y->p[j]; p < Y->p[j+1] && eq; p++)
        {
            int64_t i = Y->i[p];
            if (w[i] >= 0)
            {
                if (mpz_cmp (X->x.mpz[w[i]], Y->x.mpz[p]) != 0) eq = false;
                w[i] = -2;      // matched
            }
            else
            {
                // entry only in Y (or a duplicate): must be an explicit zero
                if (w[i] == -2 || mpz_sgn (Y->x.mpz[p]) != 0) eq = false;
            }
        }
        for (int64_t p = X->p[j]; p < X->p[j+1]; p++)
        {
            int64_t i = X->i[p];
            // entries only in X must be explicit zeros
            if (w[i] >= 0 && mpz_sgn (X->x.mpz[p]) != 0) eq = false;
            w[i] = -1;          // reset the workspace
        }
    }
    SLIP_FREE (w);
    return eq;
}

/* Prompt the user for the swap position k, 1-based, 1 <= k <= n-1, meaning
 * "swap columns k and k+1 of PAQ".  Returns k-1 (the 0-based index of the
 * left column of the pair), or -1 on invalid input.  The prompt is written to
 * stderr so it does not pollute the matrices printed to stdout.
 */
static int64_t ask_k (int64_t n)
{
    long k = 0;
    fprintf (stderr,
        "Enter k (1..%"PRId64") to swap adjacent columns k and k+1 of PAQ: ",
        n - 1);
    if (scanf ("%ld", &k) != 1 || k < 1 || k > n - 1)
    {
        fprintf (stderr, "  (invalid k; nothing to do)\n");
        return -1;
    }
    return (int64_t) (k - 1);   // 0-based left column of the adjacent pair
}

/* Prompt the user for the basis column to replace, 1-based, 1 <= c <= n.
 * Returns c-1 (0-based), or -1 on invalid input. */
static int64_t ask_col (int64_t n)
{
    long c = 0;
    fprintf (stderr, "Enter the basis column to replace with a dense column "
        "(1..%"PRId64"): ", n);
    if (scanf ("%ld", &c) != 1 || c < 1 || c > n)
    {
        fprintf (stderr, "  (invalid column; nothing to do)\n");
        return -1;
    }
    return (int64_t) (c - 1);
}

//------------------------------------------------------------------------------
// QSOpt_ex basis reader
//------------------------------------------------------------------------------

/* Read a basis matrix emitted by QSOpt_ex and convert it to the CSC MPZ
 * form SLIP LU expects.  Two on-disk layouts are supported, distinguished
 * by the 'rational' flag (set by the format sniffer in main):
 *
 *  OLD (row-scaled; 50v-10_LP_Bases/, co9_Bases/):
 *      nrows ncols nnz         (ncols == nrows for a basis)
 *      L_0 ... L_{nrows-1}     one per line: row-wise denominator LCMs
 *      row col int_val         repeated nnz times, 0-indexed; the true
 *                              entry is B[row,col] = int_val / L_row
 *
 *  NEW (rational triplets; cycle_Bases/):
 *      nrows ncols nnz
 *      row col val             repeated nnz times, 0-indexed; val is an
 *                              integer or a rational "p/q" (no spaces)
 *
 * Either way the entries are assembled as exact rationals in a triplet
 * matrix, then SLIP_matrix_copy converts to CSC MPZ by multiplying every
 * entry by ONE integer scalar -- the LCM of all the denominators (see
 * slip_expand_mpq_array) -- and recording it in A->scale.  Scaling the
 * whole matrix by a scalar is harmless here: the workflow factors the
 * resulting integer matrix, and a basis solve only needs the scale factor
 * folded back at the end (as SLIP_LU_solve does).
 */
static SLIP_info read_qsx_basis (SLIP_matrix **A_handle, FILE *file,
    bool rational, SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    SLIP_matrix *T = NULL;      // triplet MPQ assembly of the basis
    mpz_t *Lrow = NULL;         // per-row denominators (old layout only)
    mpz_t num;                  // triplet numerator scratch
    bool num_init = false;
    int64_t m = 0, n, nnz;

    *A_handle = NULL;

    #define QCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    if (fscanf (file, "%"PRId64" %"PRId64" %"PRId64, &m, &n, &nnz) != 3
        || m != n || m <= 0 || nnz < 0)
    {
        fprintf (stderr, "bad QSOpt_ex basis header (need square matrix)\n");
        return SLIP_INCORRECT_INPUT;
    }

    mpz_init (num);
    num_init = true;

    if (!rational)
    {
        Lrow = (mpz_t *) SLIP_calloc ((size_t) m, sizeof (mpz_t));
        if (Lrow == NULL) { info = SLIP_OUT_OF_MEMORY; goto done; }
        for (int64_t i = 0; i < m; i++) mpz_init (Lrow[i]);

        for (int64_t i = 0; i < m; i++)
        {
            if (gmp_fscanf (file, "%Zd", Lrow[i]) != 1
                || mpz_sgn (Lrow[i]) <= 0)
            {
                fprintf (stderr, "bad QSOpt_ex row denominator %"PRId64"\n",
                    i);
                info = SLIP_INCORRECT_INPUT;
                goto done;
            }
        }
    }

    QCK (SLIP_matrix_allocate (&T, SLIP_TRIPLET, SLIP_MPQ, m, n, nnz,
        false, true, option));
    for (int64_t k = 0; k < nnz; k++)
    {
        int64_t row, col;
        if (rational)
        {
            // %Qd accepts both "p" and "p/q"; it does not canonicalize
            if (gmp_fscanf (file, "%"PRId64" %"PRId64" %Qd", &row, &col,
                    T->x.mpq[k]) != 3
                || row < 0 || row >= m || col < 0 || col >= n
                || mpz_sgn (mpq_denref (T->x.mpq[k])) <= 0)
            {
                fprintf (stderr, "bad QSOpt_ex triplet %"PRId64"\n", k);
                info = SLIP_INCORRECT_INPUT;
                goto done;
            }
            T->i[k] = row;
            T->j[k] = col;
            mpq_canonicalize (T->x.mpq[k]);
        }
        else
        {
            if (gmp_fscanf (file, "%"PRId64" %"PRId64" %Zd", &row, &col,
                    num) != 3
                || row < 0 || row >= m || col < 0 || col >= n)
            {
                fprintf (stderr, "bad QSOpt_ex triplet %"PRId64"\n", k);
                info = SLIP_INCORRECT_INPUT;
                goto done;
            }
            T->i[k] = row;
            T->j[k] = col;
            mpq_set_num (T->x.mpq[k], num);
            mpq_set_den (T->x.mpq[k], Lrow[row]);
            mpq_canonicalize (T->x.mpq[k]);
        }
    }
    T->nz = nnz;

    QCK (SLIP_matrix_copy (A_handle, SLIP_CSC, SLIP_MPZ, T, option));

done:
    SLIP_matrix_free (&T, option);
    if (Lrow != NULL)
    {
        for (int64_t i = 0; i < m; i++) mpz_clear (Lrow[i]);
        SLIP_FREE (Lrow);
    }
    if (num_init) mpz_clear (num);
    #undef QCK
    return info;
}

//------------------------------------------------------------------------------
// D construction helper
//------------------------------------------------------------------------------

/* Build the diagonal matrix D from the pivot sequence rhos.  D is all zeros
 * except on the diagonal, where D(k,k) is the product of consecutive pivots:
 *
 *      D(k,k) = rhos[k-1] * rhos[k],   with the convention rhos[-1] = 1.
 *
 * So D(0,0) = rhos[0], D(1,1) = rhos[0]*rhos[1], D(2,2) = rhos[1]*rhos[2], ...
 * These are the pivots that make PAQ = L * D^(-1) * U.  On success *D_handle
 * points to a freshly allocated n-by-n dense MPZ matrix.
 *
 * When a 2x2 block pivot was used at positions {kb, kb+1} (kb >= 0; see
 * apdpu), D is block diagonal there instead: its 2x2 block is
 * rhos[kb-1] * B', where B' is the block the update stored in rows/columns
 * {kb, kb+1} of the factors (read here from the dense factor Lb).  This
 * generalizes the scalar D(k,k) = rhos[k-1] * B with B = [rhos[k]].  Pass
 * kb = -1 (Lb ignored) for the ordinary all-scalar D.
 */
static SLIP_info build_D (SLIP_matrix **D_handle, const SLIP_matrix *rhos,
    int64_t n, int64_t kb, const SLIP_matrix *Lb, const SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *D = NULL;

    // init=true zeros every entry, so only the diagonal needs to be filled.
    info = SLIP_matrix_allocate (&D, SLIP_DENSE, SLIP_MPZ, n, n, n*n,
        false, true, option);
    if (info != SLIP_OK) return info;

    for (int64_t k = 0; k < n; k++)
    {
        if (k == 0)
        {
            info = SLIP_mpz_set (SLIP_2D (D, k, k, mpz), rhos->x.mpz[k]);
        }
        else
        {
            info = SLIP_mpz_mul (SLIP_2D (D, k, k, mpz),
                rhos->x.mpz[k-1], rhos->x.mpz[k]);
        }
        if (info != SLIP_OK) { SLIP_matrix_free (&D, option); return info; }
    }

    if (kb >= 0)        // 2x2 block pivot: D block = rhos[kb-1] * B'
    {
        for (int64_t r = 0; r < 2 && info == SLIP_OK; r++)
        {
            for (int64_t c = 0; c < 2 && info == SLIP_OK; c++)
            {
                if (kb > 0)
                {
                    info = SLIP_mpz_mul (SLIP_2D (D, kb+r, kb+c, mpz),
                        rhos->x.mpz[kb-1], SLIP_2D (Lb, kb+r, kb+c, mpz));
                }
                else
                {
                    info = SLIP_mpz_set (SLIP_2D (D, kb+r, kb+c, mpz),
                        SLIP_2D (Lb, kb+r, kb+c, mpz));
                }
            }
        }
        if (info != SLIP_OK) { SLIP_matrix_free (&D, option); return info; }
    }

    *D_handle = D;
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// APCPU: adjacent-column push update
//------------------------------------------------------------------------------

/* Update the REF/SLIP factors after swapping ADJACENT columns kc and kc+1
 * (0-based) of the factored matrix.
 *
 * Input:  L, U, rhos -- dense factors of PAQ = L * D^(-1) * U, where
 *                       L(j,j) = U(j,j) = rhos[j] and D(j,j) =
 *                       rhos[j-1]*rhos[j] with rhos[-1] = 1.
 *         PAQ_new    -- PAQ with columns kc and kc+1 ALREADY swapped.
 * Output: *L_out, *U_out, *rhos_out -- freshly allocated dense factors of
 *         PAQ_new (the caller rebuilds D from *rhos_out with build_D).
 *
 * Everything in the factors is a subdeterminant (minor) of PAQ (Bareiss/IPGE):
 * with 0-based index sets and M(R,C) = det of the submatrix with rows R and
 * columns C in increasing order,
 *
 *      rhos[j] = M({0..j},   {0..j})            (leading (j+1)-minor)
 *      L[i,j]  = M({0..j-1, i}, {0..j})         for i >= j
 *      U[i,j]  = M({0..i},   {0..i-1, j})       for j >= i
 *
 * Swapping columns kc and kc+1 changes a minor only if its column set touches
 * the pair, and only its ORDER changes if the set contains both.  Hence:
 *
 *   pivots:  rhos'[j] = rhos[j]                       j <  kc     (untouched)
 *            rhos'[kc] = M({0..kc}, {0..kc-1, kc+1})  = new pivot, obtained
 *                        from a REF forward solve of PAQ_new(:,kc)
 *            rhos'[j] = -rhos[j]                      j >= kc+1   (both cols in
 *                        the leading set, transposed => sign flip).  Note the
 *                        flip applies to ALL j >= kc+1, so D'(j,j) =
 *                        rhos'[j-1]*rhos'[j] = D(j,j) for j >= kc+2.
 *
 *   L:  cols j < kc untouched.  Col kc is the "data" column: L'(i,kc) =
 *       M({0..kc-1, i}, {0..kc-1, kc+1}) = PAQ_new(:,kc) pushed through the
 *       first kc IPGE steps (REF forward solve; those steps use only the
 *       untouched L cols 0..kc-1 and pivots rhos[0..kc-1]).  Cols j >= kc+1:
 *       column set contains both swapped columns => sign flip, INCLUDING the
 *       diagonal (L'(j,j) = rhos'[j] = -rhos[j]).
 *
 *   U:  rows i < kc: the entries in cols kc and kc+1 trade places --
 *       U'(i,kc) = M({0..i},{0..i-1, kc+1}) = old U(i,kc+1) (= x[i] from the
 *       same forward solve) and U'(i,kc+1) = old U(i,kc); all other entries
 *       untouched.  Row kc: U'(kc,kc) = rhos'[kc]; U'(kc,kc+1) =
 *       M({0..kc},{0..kc}) = old rhos[kc]; U'(kc,j) untouched for j >= kc+2.
 *       Row kc+1: recomputed (the O(n-kc) strip, see below); diagonal =
 *       rhos'[kc+1] = -rhos[kc+1].  Rows i >= kc+2: sign flip (incl. diag).
 *
 * The row-(kc+1) strip: for j >= kc+2 the target U'(kc+1,j) =
 * M({0..kc+1}, {0..kc-1, kc+1, j}).  Sylvester's determinant identity with
 * inner minor rows/cols {0..kc-1}, extra rows {kc, kc+1}, extra cols
 * {kc+1, j} gives
 *
 *      U'(kc+1,j) = ( rhos'[kc] * xint - L'(kc+1,kc) * U(kc,j) ) / rhos[kc-1]
 *
 * where xint = M({0..kc-1, kc+1}, {0..kc-1, j}) is the step-(kc-1) IPGE value
 * of the ORIGINAL matrix at (kc+1, j).  It is not stored (old U(kc+1,j) holds
 * the step-kc value) but is recovered exactly by inverting the original
 * step-kc IPGE update:
 *
 *      xint = ( rhos[kc-1] * U(kc+1,j) + L(kc+1,kc) * U(kc,j) ) / rhos[kc]
 *
 * Both divisions are exact (every quotient is itself a minor).
 *
 * Fails with SLIP_SINGULAR if the new pivot rhos'[kc] is zero (the swapped
 * matrix then has no REF LU factorization with this row permutation).
 */
static SLIP_info apcpu (SLIP_matrix **L_out, SLIP_matrix **U_out,
    SLIP_matrix **rhos_out, SLIP_matrix *L, SLIP_matrix *U,
    SLIP_matrix *rhos, const SLIP_matrix *PAQ_new, int64_t kc,
    const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *Lp = NULL, *Up = NULL, *rp = NULL;   // the updated factors
    SLIP_matrix *x = NULL;                            // forward-solve workspace
    SLIP_matrix *tmp = NULL;                          // scratch mpz scalars

    if (kc < 0 || kc > n - 2) return SLIP_INCORRECT_INPUT;

    #define ACK(method) { info = (method); if (info != SLIP_OK) goto done; }

    // start from copies of the old factors; untouched regions stay as-is
    ACK (SLIP_matrix_copy (&Lp, SLIP_DENSE, SLIP_MPZ, L, option));
    ACK (SLIP_matrix_copy (&Up, SLIP_DENSE, SLIP_MPZ, U, option));
    ACK (SLIP_matrix_copy (&rp, SLIP_DENSE, SLIP_MPZ, rhos, option));

    ACK (SLIP_matrix_allocate (&x, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, true, option));
    ACK (SLIP_matrix_allocate (&tmp, SLIP_DENSE, SLIP_MPZ, 2, 1, 2,
        false, true, option));
    mpz_t *t1 = &SLIP_1D (tmp, 0, mpz);
    mpz_t *t2 = &SLIP_1D (tmp, 1, mpz);

    //--------------------------------------------------------------------------
    // Steps 1 & 2: REF forward solve of PAQ_new(:,kc) through pivots
    // 0..kc-1.  Afterwards x[i] = U'(i,kc) for i < kc, x[kc] = rhos'[kc],
    // and x[i] = L'(i,kc) for i > kc.
    //--------------------------------------------------------------------------

    for (int64_t i = 0; i < n; i++)
    {
        ACK (SLIP_mpz_set (SLIP_1D (x, i, mpz), SLIP_2D (PAQ_new, i, kc, mpz)));
    }
    for (int64_t t = 0; t < kc; t++)
    {
        for (int64_t i = t + 1; i < n; i++)
        {
            // x[i] = ( rhos[t]*x[i] - L(i,t)*x[t] ) / rhos[t-1]
            ACK (SLIP_mpz_mul (*t1, rhos->x.mpz[t], SLIP_1D (x, i, mpz)));
            ACK (SLIP_mpz_submul (*t1, SLIP_2D (L, i, t, mpz),
                SLIP_1D (x, t, mpz)));
            if (t > 0)
            {
                ACK (SLIP_mpz_divexact (SLIP_1D (x, i, mpz), *t1,
                    rhos->x.mpz[t-1]));
            }
            else
            {
                ACK (SLIP_mpz_set (SLIP_1D (x, i, mpz), *t1));
            }
        }
    }

    // new pivot sequence: rhos'[kc] from the solve, sign flip for j >= kc+1
    int sgn;
    ACK (SLIP_mpz_sgn (&sgn, SLIP_1D (x, kc, mpz)));
    if (sgn == 0) { info = SLIP_SINGULAR; goto done; }
    ACK (SLIP_mpz_set (rp->x.mpz[kc], SLIP_1D (x, kc, mpz)));
    for (int64_t j = kc + 1; j < n; j++)
    {
        mpz_neg (rp->x.mpz[j], rhos->x.mpz[j]);
    }

    // scatter x into the factors: U'(0..kc-1, kc) and L'(kc..n-1, kc)
    for (int64_t i = 0; i < kc; i++)
    {
        ACK (SLIP_mpz_set (SLIP_2D (Up, i, kc, mpz), SLIP_1D (x, i, mpz)));
    }
    for (int64_t i = kc; i < n; i++)
    {
        ACK (SLIP_mpz_set (SLIP_2D (Lp, i, kc, mpz), SLIP_1D (x, i, mpz)));
    }

    //--------------------------------------------------------------------------
    // Step 3: column kc+1 of U in rows 0..kc gets the OLD column kc entries
    // (U'(i,kc+1) = old U(i,kc); in particular U'(kc,kc+1) = old pivot
    // rhos[kc]), and the diagonal U'(kc,kc) becomes the new pivot.
    //--------------------------------------------------------------------------

    for (int64_t i = 0; i <= kc; i++)
    {
        ACK (SLIP_mpz_set (SLIP_2D (Up, i, kc+1, mpz), SLIP_2D (U, i, kc, mpz)));
    }
    ACK (SLIP_mpz_set (SLIP_2D (Up, kc, kc, mpz), rp->x.mpz[kc]));
    // U'(kc, j) for j >= kc+2 is untouched (already in the copy)

    //--------------------------------------------------------------------------
    // Step 4: sign flips on the trailing block -- every stored entry whose
    // defining column set contains BOTH swapped columns.  For L that is
    // every column j >= kc+1 (diagonal included); for U every row
    // i >= kc+2 (diagonal included).  Row kc+1 of U is rebuilt in Step 5.
    //--------------------------------------------------------------------------

    for (int64_t j = kc + 1; j < n; j++)
    {
        for (int64_t i = j; i < n; i++)
        {
            mpz_neg (SLIP_2D (Lp, i, j, mpz), SLIP_2D (Lp, i, j, mpz));
        }
    }
    for (int64_t i = kc + 2; i < n; i++)
    {
        for (int64_t j = i; j < n; j++)
        {
            mpz_neg (SLIP_2D (Up, i, j, mpz), SLIP_2D (Up, i, j, mpz));
        }
    }

    //--------------------------------------------------------------------------
    // Step 5: recompute row kc+1 of U for j >= kc+2 (the O(n-kc) strip) via
    // the Sylvester identity described above; the diagonal is the new pivot.
    //--------------------------------------------------------------------------

    ACK (SLIP_mpz_set (SLIP_2D (Up, kc+1, kc+1, mpz), rp->x.mpz[kc+1]));
    for (int64_t j = kc + 2; j < n; j++)
    {
        // xint = ( rhos[kc-1]*U(kc+1,j) + L(kc+1,kc)*U(kc,j) ) / rhos[kc]
        if (kc > 0)
        {
            ACK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1],
                SLIP_2D (U, kc+1, j, mpz)));
        }
        else
        {
            ACK (SLIP_mpz_set (*t1, SLIP_2D (U, kc+1, j, mpz)));
        }
        mpz_addmul (*t1, SLIP_2D (L, kc+1, kc, mpz),
            SLIP_2D (U, kc, j, mpz));
        ACK (SLIP_mpz_divexact (*t1, *t1, rhos->x.mpz[kc]));

        // U'(kc+1,j) = ( rhos'[kc]*xint - L'(kc+1,kc)*U(kc,j) ) / rhos[kc-1]
        ACK (SLIP_mpz_mul (*t2, rp->x.mpz[kc], *t1));
        ACK (SLIP_mpz_submul (*t2, SLIP_2D (Lp, kc+1, kc, mpz),
            SLIP_2D (U, kc, j, mpz)));
        if (kc > 0)
        {
            ACK (SLIP_mpz_divexact (SLIP_2D (Up, kc+1, j, mpz), *t2,
                rhos->x.mpz[kc-1]));
        }
        else
        {
            ACK (SLIP_mpz_set (SLIP_2D (Up, kc+1, j, mpz), *t2));
        }
    }

    // hand the updated factors to the caller
    *L_out = Lp;       Lp = NULL;
    *U_out = Up;       Up = NULL;
    *rhos_out = rp;    rp = NULL;

done:
    SLIP_matrix_free (&Lp, option);
    SLIP_matrix_free (&Up, option);
    SLIP_matrix_free (&rp, option);
    SLIP_matrix_free (&x, option);
    SLIP_matrix_free (&tmp, option);
    #undef ACK
    return info;
}

//------------------------------------------------------------------------------
// APDPU: adjacent diagonal (symmetric) push update, dense
//------------------------------------------------------------------------------

/* Update the dense factors after the SYMMETRIC swap PAQ' = E*PAQ*E, where E
 * exchanges positions kc and kc+1 (0-based): both columns kc,kc+1 AND rows
 * kc,kc+1 of PAQ are swapped (the row swap is exactly a column push applied
 * to the transpose).
 *
 * With M(R,C) as in apcpu above, the symmetric swap gives
 * M'(R,C) = s(R)*s(C)*M(sigma R, sigma C), s(X) = -1 iff {kc,kc+1} in X.
 * Unlike the one-sided column push there are NO sign flips: wherever both
 * indices sit in R and C the two signs cancel (the trailing block
 * i,j >= kc+2 is fully invariant), and wherever exactly one appears the
 * minor maps to another minor -- either one already stored (the corner
 * block and the row/column swaps below) or one recoverable with a single
 * exact division (the four strips).  PAQ itself is never consulted:
 *
 *   pivot (Sylvester identity, also the O(1) feasibility test):
 *     rho'[kc] = ( rho[kc-1]*rho[kc+1] + L(kc+1,kc)*U(kc,kc+1) ) / rho[kc]
 *   corner block {kc,kc+1} x {kc,kc+1}:
 *     L'(kc,kc) = U'(kc,kc) = rho'[kc];   L'(kc+1,kc) = old U(kc,kc+1);
 *     U'(kc,kc+1) = old L(kc+1,kc);       (kc+1,kc+1) diagonals unchanged
 *   stored-value swaps:
 *     rows kc,kc+1 of L trade places in columns < kc;
 *     columns kc,kc+1 of U trade places in rows < kc
 *   four strips for i,j >= kc+2 (backtrack at rho[kc], then forward IPGE
 *   at the new pivot rho'[kc]):
 *     L'(i,kc)   = ( rho[kc-1]*L(i,kc+1) + L(i,kc)*U(kc,kc+1) ) / rho[kc]
 *     L'(i,kc+1) = ( rho'[kc]*L(i,kc) - L'(i,kc)*L(kc+1,kc) ) / rho[kc-1]
 *     U'(kc,j)   = ( rho[kc-1]*U(kc+1,j) + L(kc+1,kc)*U(kc,j) ) / rho[kc]
 *     U'(kc+1,j) = ( rho'[kc]*U(kc,j) - U(kc,kc+1)*U'(kc,j) ) / rho[kc-1]
 *
 * rho[-1] = 1 and every division is exact (each quotient is a minor of
 * PAQ').  All other pivots are unchanged, so D changes only at positions
 * kc and kc+1 (the caller rebuilds it from rhos').
 *
 * ZERO PIVOT => 2x2 BLOCK PIVOT FALLBACK.  rho'[kc] = 0 means the leading
 * (kc+1)-minor of PAQ' vanishes, so no scalar REF LU with this row order
 * exists (PAQ' itself is NOT singular: det PAQ' = +/- rho[n-1]).  If
 * used_block is non-NULL, positions kc and kc+1 are then eliminated
 * TOGETHER as a 2x2 block pivot (Bunch-Kaufman style), which cannot fail:
 * the block determinant is rho[kc-1]*rho[kc+1] != 0.  Rows/columns
 * {kc,kc+1} of the factors then hold step-(kc-1) values throughout:
 *
 *   block (stored in BOTH L' and U', generalizing the scalar diagonal):
 *     B' = [ rho'[kc] (= 0)    old L(kc+1,kc) ]
 *          [ old U(kc,kc+1)    old rho[kc]    ]
 *   strips (i,j >= kc+2): the same backtracks as 6a/6c above, but NO
 *   forward IPGE step -- the partner strip is a verbatim copy:
 *     L'(i,kc) = backtrack;   L'(i,kc+1) = old L(i,kc)
 *     U'(kc,j) = backtrack;   U'(kc+1,j) = old U(kc,j)
 *
 * rhos'[kc] stays 0 (the block marker), D' gets the 2x2 block
 * rho[kc-1]*B' (see build_D), and PAQ' = L' * D'^(-1) * U' still holds
 * with the true 2x2 inverse.  *used_block reports which form was built.
 * If used_block is NULL, a zero pivot fails with SLIP_SINGULAR as before.
 */
static SLIP_info apdpu (SLIP_matrix **L_out, SLIP_matrix **U_out,
    SLIP_matrix **rhos_out, SLIP_matrix *L, SLIP_matrix *U,
    SLIP_matrix *rhos, int64_t kc, bool *used_block,
    const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *Lp = NULL, *Up = NULL, *rp = NULL;   // the updated factors
    SLIP_matrix *tmp = NULL;                          // scratch mpz scalar

    if (kc < 0 || kc > n - 2) return SLIP_INCORRECT_INPUT;

    #define DCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    // start from copies; the trailing block and everything left of / above
    // the two swap positions is already correct
    DCK (SLIP_matrix_copy (&Lp, SLIP_DENSE, SLIP_MPZ, L, option));
    DCK (SLIP_matrix_copy (&Up, SLIP_DENSE, SLIP_MPZ, U, option));
    DCK (SLIP_matrix_copy (&rp, SLIP_DENSE, SLIP_MPZ, rhos, option));
    DCK (SLIP_matrix_allocate (&tmp, SLIP_DENSE, SLIP_MPZ, 1, 1, 1,
        false, true, option));
    mpz_t *t1 = &SLIP_1D (tmp, 0, mpz);

    //--------------------------------------------------------------------------
    // Step 1: new pivot rho'[kc] via Sylvester; all other pivots unchanged
    //--------------------------------------------------------------------------

    if (kc > 0)
    {
        DCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1], rhos->x.mpz[kc+1]));
    }
    else
    {
        DCK (SLIP_mpz_set (*t1, rhos->x.mpz[kc+1]));
    }
    mpz_addmul (*t1, SLIP_2D (L, kc+1, kc, mpz), SLIP_2D (U, kc, kc+1, mpz));
    DCK (SLIP_mpz_divexact (rp->x.mpz[kc], *t1, rhos->x.mpz[kc]));

    int sgn;
    DCK (SLIP_mpz_sgn (&sgn, rp->x.mpz[kc]));
    bool blk = (sgn == 0);          // 2x2 block pivot fallback engaged
    if (blk && used_block == NULL) { info = SLIP_SINGULAR; goto done; }
    if (used_block != NULL) *used_block = blk;

    //--------------------------------------------------------------------------
    // Step 3: corner block {kc,kc+1} x {kc,kc+1}, all from storage
    //--------------------------------------------------------------------------

    DCK (SLIP_mpz_set (SLIP_2D (Lp, kc, kc, mpz), rp->x.mpz[kc]));
    DCK (SLIP_mpz_set (SLIP_2D (Up, kc, kc, mpz), rp->x.mpz[kc]));
    DCK (SLIP_mpz_set (SLIP_2D (Lp, kc+1, kc, mpz),
        SLIP_2D (U, kc, kc+1, mpz)));
    DCK (SLIP_mpz_set (SLIP_2D (Up, kc, kc+1, mpz),
        SLIP_2D (L, kc+1, kc, mpz)));
    // the (kc+1,kc+1) diagonals keep rho[kc+1] -- already in the copies
    if (blk)
    {
        // full block B' in BOTH factors: mirror entries and rho[kc] on the
        // trailing diagonal instead of rho[kc+1]
        DCK (SLIP_mpz_set (SLIP_2D (Lp, kc, kc+1, mpz),
            SLIP_2D (L, kc+1, kc, mpz)));
        DCK (SLIP_mpz_set (SLIP_2D (Up, kc+1, kc, mpz),
            SLIP_2D (U, kc, kc+1, mpz)));
        DCK (SLIP_mpz_set (SLIP_2D (Lp, kc+1, kc+1, mpz), rhos->x.mpz[kc]));
        DCK (SLIP_mpz_set (SLIP_2D (Up, kc+1, kc+1, mpz), rhos->x.mpz[kc]));
    }

    //--------------------------------------------------------------------------
    // Steps 4 & 5: stored-value swaps in the leading rows/columns
    //--------------------------------------------------------------------------

    for (int64_t j = 0; j < kc; j++)        // rows kc,kc+1 of L, cols < kc
    {
        mpz_swap (SLIP_2D (Lp, kc, j, mpz), SLIP_2D (Lp, kc+1, j, mpz));
    }
    for (int64_t i = 0; i < kc; i++)        // cols kc,kc+1 of U, rows < kc
    {
        mpz_swap (SLIP_2D (Up, i, kc, mpz), SLIP_2D (Up, i, kc+1, mpz));
    }

    //--------------------------------------------------------------------------
    // Step 6: the four strips (all reads from the ORIGINAL L, U except the
    // just-written backtrack values, so no ordering hazards)
    //--------------------------------------------------------------------------

    for (int64_t i = kc + 2; i < n; i++)    // 6a/6b: columns kc, kc+1 of L
    {
        // 6a: backtrack -- L'(i,kc) = a^{kc-1}(i,kc+1) of the old matrix
        if (kc > 0)
        {
            DCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1],
                SLIP_2D (L, i, kc+1, mpz)));
        }
        else
        {
            DCK (SLIP_mpz_set (*t1, SLIP_2D (L, i, kc+1, mpz)));
        }
        mpz_addmul (*t1, SLIP_2D (L, i, kc, mpz), SLIP_2D (U, kc, kc+1, mpz));
        DCK (SLIP_mpz_divexact (SLIP_2D (Lp, i, kc, mpz), *t1,
            rhos->x.mpz[kc]));

        if (blk)
        {
            // block: L'(i,kc+1) = a'^{kc-1}(i,kc+1) = old L(i,kc), verbatim
            DCK (SLIP_mpz_set (SLIP_2D (Lp, i, kc+1, mpz),
                SLIP_2D (L, i, kc, mpz)));
            continue;
        }

        // 6b: forward IPGE at the new pivot rho'[kc]
        DCK (SLIP_mpz_mul (*t1, rp->x.mpz[kc], SLIP_2D (L, i, kc, mpz)));
        DCK (SLIP_mpz_submul (*t1, SLIP_2D (Lp, i, kc, mpz),
            SLIP_2D (L, kc+1, kc, mpz)));
        if (kc > 0)
        {
            DCK (SLIP_mpz_divexact (SLIP_2D (Lp, i, kc+1, mpz), *t1,
                rhos->x.mpz[kc-1]));
        }
        else
        {
            DCK (SLIP_mpz_set (SLIP_2D (Lp, i, kc+1, mpz), *t1));
        }
    }

    for (int64_t j = kc + 2; j < n; j++)    // 6c/6d: rows kc, kc+1 of U
    {
        // 6c: backtrack -- U'(kc,j) = a^{kc-1}(kc+1,j) of the old matrix
        if (kc > 0)
        {
            DCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1],
                SLIP_2D (U, kc+1, j, mpz)));
        }
        else
        {
            DCK (SLIP_mpz_set (*t1, SLIP_2D (U, kc+1, j, mpz)));
        }
        mpz_addmul (*t1, SLIP_2D (L, kc+1, kc, mpz), SLIP_2D (U, kc, j, mpz));
        DCK (SLIP_mpz_divexact (SLIP_2D (Up, kc, j, mpz), *t1,
            rhos->x.mpz[kc]));

        if (blk)
        {
            // block: U'(kc+1,j) = a'^{kc-1}(kc+1,j) = old U(kc,j), verbatim
            DCK (SLIP_mpz_set (SLIP_2D (Up, kc+1, j, mpz),
                SLIP_2D (U, kc, j, mpz)));
            continue;
        }

        // 6d: forward IPGE at the new pivot rho'[kc]
        DCK (SLIP_mpz_mul (*t1, rp->x.mpz[kc], SLIP_2D (U, kc, j, mpz)));
        DCK (SLIP_mpz_submul (*t1, SLIP_2D (U, kc, kc+1, mpz),
            SLIP_2D (Up, kc, j, mpz)));
        if (kc > 0)
        {
            DCK (SLIP_mpz_divexact (SLIP_2D (Up, kc+1, j, mpz), *t1,
                rhos->x.mpz[kc-1]));
        }
        else
        {
            DCK (SLIP_mpz_set (SLIP_2D (Up, kc+1, j, mpz), *t1));
        }
    }

    // Step 7: trailing block unchanged; Step 8 (D) is rebuilt by the caller

    *L_out = Lp;      Lp = NULL;
    *U_out = Up;      Up = NULL;
    *rhos_out = rp;   rp = NULL;

done:
    SLIP_matrix_free (&Lp, option);
    SLIP_matrix_free (&Up, option);
    SLIP_matrix_free (&rp, option);
    SLIP_matrix_free (&tmp, option);
    #undef DCK
    return info;
}

//------------------------------------------------------------------------------
// sparse APCPU: adjacent-column push update on the CSC factors
//------------------------------------------------------------------------------

/* Update the CSC factors of PAQ (as returned by SLIP_LU_factorize: L lower
 * CSC with column j holding rows i >= j, U upper CSC with column j holding
 * rows i <= j, both including the pivot diagonal, row indices in PAQ space)
 * to the factors of PAQ' with adjacent columns kc and kc+1 (0-based)
 * swapped.
 *
 * Same minor algebra as the dense apcpu() above, but the two "data" slices
 * are obtained with the BACK-SOLVE TRICK -- inverting one stored IPGE step
 * instead of replaying kc of them -- so the update touches only stored
 * entries of L, U, and rhos.  PAQ itself is never accessed, and the work is
 * O(nnz of the affected columns/rows), independent of kc:
 *
 *   new L column kc (i >= kc+1; also gives the diagonal = rho'[kc]):
 *      L'(i,kc) = ( rho[kc-1]*L(i,kc+1) + L(i,kc)*U(kc,kc+1) ) / rho[kc]
 *   new U row kc+1 (j >= kc+2), via the unstored intermediate xint:
 *      xint       = ( rho[kc-1]*U(kc+1,j) + L(kc+1,kc)*U(kc,j) ) / rho[kc]
 *      U'(kc+1,j) = ( rho'[kc]*xint - L'(kc+1,kc)*U(kc,j) ) / rho[kc-1]
 *
 * Absent entries are zeros, rho[-1] = 1, and every division is exact (each
 * quotient is itself a minor of PAQ).  All other stored entries either move
 * (the two swapped columns of U in rows <= kc), flip sign (L columns
 * >= kc+1 and U rows >= kc+2, diagonals included), or are untouched.
 *
 * The new factors can gain entries relative to the old pattern (fill-in in
 * L column kc and U row kc+1), so fresh matrices are allocated and returned
 * through L_out / U_out / rhos_out.  Fails with SLIP_SINGULAR if the new
 * pivot rho'[kc] = U(kc,kc+1) is absent or zero.
 */
static SLIP_info apcpu_sparse (SLIP_matrix **L_out, SLIP_matrix **U_out,
    SLIP_matrix **rhos_out, SLIP_matrix *L, SLIP_matrix *U,
    SLIP_matrix *rhos, int64_t kc, const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *Lp = NULL, *Up = NULL, *rp = NULL;   // the updated factors
    SLIP_matrix *tmp = NULL;                          // scratch mpz scalars
    int64_t *w = NULL;                                // scatter workspace

    if (kc < 0 || kc > n - 2) return SLIP_INCORRECT_INPUT;

    #define SCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    //--------------------------------------------------------------------------
    // locate the stored entries the identities need
    //--------------------------------------------------------------------------

    // uk_pos: position of U(kc,kc+1), the new pivot rho'[kc]
    int64_t uk_pos = -1;
    for (int64_t p = U->p[kc+1]; p < U->p[kc+2]; p++)
    {
        if (U->i[p] == kc) { uk_pos = p; break; }
    }
    int sgn = 0;
    if (uk_pos >= 0) SCK (SLIP_mpz_sgn (&sgn, U->x.mpz[uk_pos]));
    if (uk_pos < 0 || sgn == 0) { info = SLIP_SINGULAR; goto done; }

    // lk_old_pos: position of L(kc+1,kc) in the old column kc (-1 if zero)
    int64_t lk_old_pos = -1;
    for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++)
    {
        if (L->i[p] == kc + 1) { lk_old_pos = p; break; }
    }

    //--------------------------------------------------------------------------
    // workspace and the new pivot sequence
    //--------------------------------------------------------------------------

    w = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (w == NULL) { info = SLIP_OUT_OF_MEMORY; goto done; }
    for (int64_t i = 0; i < n; i++) w[i] = -1;

    SCK (SLIP_matrix_allocate (&tmp, SLIP_DENSE, SLIP_MPZ, 2, 1, 2,
        false, true, option));
    mpz_t *t1 = &SLIP_1D (tmp, 0, mpz);
    mpz_t *t2 = &SLIP_1D (tmp, 1, mpz);

    SCK (SLIP_matrix_copy (&rp, SLIP_DENSE, SLIP_MPZ, rhos, option));
    SCK (SLIP_mpz_set (rp->x.mpz[kc], U->x.mpz[uk_pos]));
    for (int64_t j = kc + 1; j < n; j++)
    {
        mpz_neg (rp->x.mpz[j], rhos->x.mpz[j]);
    }

    //--------------------------------------------------------------------------
    // build the new L
    //--------------------------------------------------------------------------

    // worst case: column kc grows by the pattern of column kc+1 plus 1
    SCK (SLIP_matrix_allocate (&Lp, SLIP_CSC, SLIP_MPZ, n, n,
        L->p[n] + (L->p[kc+2] - L->p[kc+1]) + 2, false, true, option));

    int64_t lnz = 0;
    for (int64_t j = 0; j < kc; j++)            // columns < kc: verbatim
    {
        Lp->p[j] = lnz;
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            Lp->i[lnz] = L->i[p];
            SCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
            lnz++;
        }
    }

    // column kc: diagonal = rho'[kc]; off-diagonals by the back-solve merge
    // over the union of the old column-kc and column-kc+1 patterns
    Lp->p[kc] = lnz;
    Lp->i[lnz] = kc;
    SCK (SLIP_mpz_set (Lp->x.mpz[lnz], U->x.mpz[uk_pos]));
    lnz++;
    for (int64_t p = L->p[kc+1]; p < L->p[kc+2]; p++)
    {
        // scatter rho[kc-1] * L(i,kc+1); includes i = kc+1 whose stored
        // "diagonal" value L(kc+1,kc+1) = rho[kc+1] belongs in the formula
        int64_t i = L->i[p];                    // i >= kc+1
        w[i] = lnz;
        Lp->i[lnz] = i;
        if (kc > 0)
        {
            SCK (SLIP_mpz_mul (Lp->x.mpz[lnz], rhos->x.mpz[kc-1],
                L->x.mpz[p]));
        }
        else
        {
            SCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
        }
        lnz++;
    }
    for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++)
    {
        // add L(i,kc) * U(kc,kc+1)
        int64_t i = L->i[p];
        if (i == kc) continue;                  // old diagonal not in formula
        if (w[i] >= 0)
        {
            mpz_addmul (Lp->x.mpz[w[i]], L->x.mpz[p], U->x.mpz[uk_pos]);
        }
        else
        {
            w[i] = lnz;
            Lp->i[lnz] = i;
            SCK (SLIP_mpz_mul (Lp->x.mpz[lnz], L->x.mpz[p],
                U->x.mpz[uk_pos]));
            lnz++;
        }
    }
    for (int64_t p = Lp->p[kc] + 1; p < lnz; p++)   // exact division
    {
        SCK (SLIP_mpz_divexact (Lp->x.mpz[p], Lp->x.mpz[p],
            rhos->x.mpz[kc]));
    }
    int64_t lk_new_pos = w[kc+1];               // L'(kc+1,kc), -1 if zero
    for (int64_t p = Lp->p[kc] + 1; p < lnz; p++)   // reset the workspace
    {
        w[Lp->i[p]] = -1;
    }

    for (int64_t j = kc + 1; j < n; j++)        // columns > kc: sign flip
    {
        Lp->p[j] = lnz;
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            Lp->i[lnz] = L->i[p];
            mpz_neg (Lp->x.mpz[lnz], L->x.mpz[p]);
            lnz++;
        }
    }
    Lp->p[n] = lnz;

    //--------------------------------------------------------------------------
    // build the new U
    //--------------------------------------------------------------------------

    // worst case: one new row-(kc+1) entry in every column >= kc+2, plus the
    // new diagonal of column kc+1
    SCK (SLIP_matrix_allocate (&Up, SLIP_CSC, SLIP_MPZ, n, n,
        U->p[n] + (n - kc) + 2, false, true, option));

    int64_t unz = 0;
    for (int64_t j = 0; j < kc; j++)            // columns < kc: verbatim
    {
        Up->p[j] = unz;
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            Up->i[unz] = U->i[p];
            SCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
            unz++;
        }
    }

    // column kc: old column kc+1 without its diagonal; the old (kc,kc+1)
    // entry becomes the new diagonal rho'[kc]
    Up->p[kc] = unz;
    for (int64_t p = U->p[kc+1]; p < U->p[kc+2]; p++)
    {
        int64_t i = U->i[p];
        if (i == kc + 1) continue;              // old diagonal dropped
        Up->i[unz] = i;
        SCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
        unz++;
    }

    // column kc+1: all of old column kc (its diagonal rho[kc] becomes
    // U'(kc,kc+1)) plus the new diagonal rho'[kc+1] = -rho[kc+1]
    Up->p[kc+1] = unz;
    for (int64_t p = U->p[kc]; p < U->p[kc+1]; p++)
    {
        Up->i[unz] = U->i[p];
        SCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
        unz++;
    }
    Up->i[unz] = kc + 1;
    SCK (SLIP_mpz_set (Up->x.mpz[unz], rp->x.mpz[kc+1]));
    unz++;

    // columns >= kc+2: rows <= kc untouched, row kc+1 recomputed by the
    // strip identities, rows >= kc+2 sign-flipped
    for (int64_t j = kc + 2; j < n; j++)
    {
        Up->p[j] = unz;

        // pass 1: locate U(kc,j) and U(kc+1,j) in the old column
        int64_t pk = -1, pk1 = -1;
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            if (U->i[p] == kc) pk = p;
            else if (U->i[p] == kc + 1) pk1 = p;
        }

        // strip value into t2 (both terms zero => no row-(kc+1) entry)
        bool have = (pk >= 0 || pk1 >= 0);
        if (have)
        {
            if (pk1 >= 0)
            {
                if (kc > 0)
                {
                    SCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1],
                        U->x.mpz[pk1]));
                }
                else
                {
                    SCK (SLIP_mpz_set (*t1, U->x.mpz[pk1]));
                }
            }
            else
            {
                SCK (SLIP_mpz_set_ui (*t1, 0));
            }
            if (pk >= 0 && lk_old_pos >= 0)
            {
                mpz_addmul (*t1, L->x.mpz[lk_old_pos], U->x.mpz[pk]);
            }
            SCK (SLIP_mpz_divexact (*t1, *t1, rhos->x.mpz[kc]));
            SCK (SLIP_mpz_mul (*t2, U->x.mpz[uk_pos], *t1));
            if (pk >= 0 && lk_new_pos >= 0)
            {
                SCK (SLIP_mpz_submul (*t2, Lp->x.mpz[lk_new_pos],
                    U->x.mpz[pk]));
            }
            if (kc > 0)
            {
                SCK (SLIP_mpz_divexact (*t2, *t2, rhos->x.mpz[kc-1]));
            }
        }

        // pass 2: emit the column
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            int64_t i = U->i[p];
            Up->i[unz] = i;
            if (i <= kc)
            {
                SCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
            }
            else if (i == kc + 1)
            {
                SCK (SLIP_mpz_set (Up->x.mpz[unz], *t2));
            }
            else
            {
                mpz_neg (Up->x.mpz[unz], U->x.mpz[p]);
            }
            unz++;
        }
        if (have && pk1 < 0)                    // fill-in in row kc+1
        {
            SCK (SLIP_mpz_sgn (&sgn, *t2));
            if (sgn != 0)
            {
                Up->i[unz] = kc + 1;
                SCK (SLIP_mpz_set (Up->x.mpz[unz], *t2));
                unz++;
            }
        }
    }
    Up->p[n] = unz;

    // hand the updated factors to the caller
    *L_out = Lp;      Lp = NULL;
    *U_out = Up;      Up = NULL;
    *rhos_out = rp;   rp = NULL;

done:
    SLIP_matrix_free (&Lp, option);
    SLIP_matrix_free (&Up, option);
    SLIP_matrix_free (&rp, option);
    SLIP_matrix_free (&tmp, option);
    SLIP_FREE (w);
    #undef SCK
    return info;
}

//------------------------------------------------------------------------------
// sparse APDPU: adjacent diagonal (symmetric) push update on the CSC factors
//------------------------------------------------------------------------------

/* Sparse version of apdpu() above: update the CSC factors of PAQ (storage
 * conventions as in apcpu_sparse) to those of PAQ' = E*PAQ*E, E swapping
 * positions kc and kc+1 (0-based).  All values come from stored entries --
 * PAQ is never consulted -- via the identities documented at apdpu():
 * the Sylvester pivot formula, the closed-form corner block, the
 * stored-value swaps (pure row-index relabels / entry moves in CSC), and
 * the four strips (backtrack at rho[kc], then forward IPGE at rho'[kc]).
 *
 * Sparsity specifics:
 *  - L columns < kc: row indices kc and kc+1 are relabeled to each other
 *    (values untouched).  U columns < kc and both factors' trailing
 *    columns >= kc+2 (except U rows kc, kc+1) are copied verbatim.
 *  - The new L column kc is built over the UNION of the old columns kc and
 *    kc+1 patterns (rows >= kc+2); entries in the union are stored even
 *    when a cancellation makes them zero, because the forward strip for
 *    column kc+1 must still visit that row (a row absent from the
 *    backtrack result can be nonzero in the forward result).
 *  - Fill-in can appear in L columns kc, kc+1 and U rows kc, kc+1, so
 *    fresh matrices are allocated and returned.
 *
 * rho'[kc] = 0 (detected in O(nnz of two columns), before anything is
 * built) triggers the 2x2 BLOCK PIVOT fallback documented at apdpu() when
 * used_block is non-NULL (SLIP_SINGULAR otherwise).  Sparse specifics of
 * the block form: the zero (kc,kc) "diagonal" is simply not stored; L
 * column kc+1 becomes B'(kc,kc+1) = old L(kc+1,kc), diagonal old rho[kc],
 * then the old column kc verbatim (no forward strip, so no fill-in there);
 * U column kc gains the below-diagonal entry B'(kc+1,kc) = old U(kc,kc+1);
 * and the trailing row kc+1 of U is the old row kc moved down.  Both
 * couplings are guaranteed present in the block case, since
 * L(kc+1,kc)*U(kc,kc+1) = -rho[kc-1]*rho[kc+1] != 0.
 */
static SLIP_info apdpu_sparse (SLIP_matrix **L_out, SLIP_matrix **U_out,
    SLIP_matrix **rhos_out, SLIP_matrix *L, SLIP_matrix *U,
    SLIP_matrix *rhos, int64_t kc, bool *used_block,
    const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *Lp = NULL, *Up = NULL, *rp = NULL;   // the updated factors
    SLIP_matrix *tmp = NULL;                          // scratch mpz scalars
    int64_t *w = NULL;                                // scatter workspace

    if (kc < 0 || kc > n - 2) return SLIP_INCORRECT_INPUT;

    #define PCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    //--------------------------------------------------------------------------
    // locate the stored couplings and compute the new pivot (Step 1)
    //--------------------------------------------------------------------------

    int64_t uk_pos = -1;                    // U(kc,kc+1), -1 if zero
    for (int64_t p = U->p[kc+1]; p < U->p[kc+2]; p++)
    {
        if (U->i[p] == kc) { uk_pos = p; break; }
    }
    int64_t lk_pos = -1;                    // L(kc+1,kc), -1 if zero
    for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++)
    {
        if (L->i[p] == kc + 1) { lk_pos = p; break; }
    }

    w = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (w == NULL) { info = SLIP_OUT_OF_MEMORY; goto done; }
    for (int64_t i = 0; i < n; i++) w[i] = -1;

    PCK (SLIP_matrix_allocate (&tmp, SLIP_DENSE, SLIP_MPZ, 2, 1, 2,
        false, true, option));
    mpz_t *t1 = &SLIP_1D (tmp, 0, mpz);
    mpz_t *t2 = &SLIP_1D (tmp, 1, mpz);

    PCK (SLIP_matrix_copy (&rp, SLIP_DENSE, SLIP_MPZ, rhos, option));
    // rho'[kc] = ( rho[kc-1]*rho[kc+1] + L(kc+1,kc)*U(kc,kc+1) ) / rho[kc]
    if (kc > 0)
    {
        PCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1], rhos->x.mpz[kc+1]));
    }
    else
    {
        PCK (SLIP_mpz_set (*t1, rhos->x.mpz[kc+1]));
    }
    if (uk_pos >= 0 && lk_pos >= 0)
    {
        mpz_addmul (*t1, L->x.mpz[lk_pos], U->x.mpz[uk_pos]);
    }
    PCK (SLIP_mpz_divexact (rp->x.mpz[kc], *t1, rhos->x.mpz[kc]));

    int sgn;
    PCK (SLIP_mpz_sgn (&sgn, rp->x.mpz[kc]));
    bool blk = (sgn == 0);          // 2x2 block pivot fallback engaged
    if (blk && used_block == NULL) { info = SLIP_SINGULAR; goto done; }
    if (used_block != NULL) *used_block = blk;
    // all other pivots are unchanged (already in the copy); in the block
    // case rho'[kc] stays 0 as the block marker, and uk_pos/lk_pos >= 0 is
    // guaranteed (their product equals -rho[kc-1]*rho[kc+1] != 0)

    //--------------------------------------------------------------------------
    // build the new L
    //--------------------------------------------------------------------------

    PCK (SLIP_matrix_allocate (&Lp, SLIP_CSC, SLIP_MPZ, n, n,
        L->p[n] + (L->p[kc+2] - L->p[kc]) + 4, false, true, option));

    int64_t lnz = 0;
    for (int64_t j = 0; j < kc; j++)    // cols < kc: relabel rows kc <-> kc+1
    {
        Lp->p[j] = lnz;
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            int64_t i = L->i[p];
            Lp->i[lnz] = (i == kc) ? kc + 1 : ((i == kc + 1) ? kc : i);
            PCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
            lnz++;
        }
    }

    // column kc: diagonal rho'[kc] (omitted in block mode, where it is 0);
    // corner L'(kc+1,kc) = old U(kc,kc+1); backtrack strip over the union
    // of the old kc / kc+1 patterns
    Lp->p[kc] = lnz;
    if (!blk)
    {
        Lp->i[lnz] = kc;
        PCK (SLIP_mpz_set (Lp->x.mpz[lnz], rp->x.mpz[kc]));
        lnz++;
    }
    if (uk_pos >= 0)
    {
        Lp->i[lnz] = kc + 1;
        PCK (SLIP_mpz_set (Lp->x.mpz[lnz], U->x.mpz[uk_pos]));
        lnz++;
    }
    int64_t strip_start = lnz;
    for (int64_t p = L->p[kc+1]; p < L->p[kc+2]; p++)
    {
        // scatter rho[kc-1] * L(i,kc+1) for i >= kc+2
        int64_t i = L->i[p];
        if (i == kc + 1) continue;              // old diagonal: corner case
        w[i] = lnz;
        Lp->i[lnz] = i;
        if (kc > 0)
        {
            PCK (SLIP_mpz_mul (Lp->x.mpz[lnz], rhos->x.mpz[kc-1],
                L->x.mpz[p]));
        }
        else
        {
            PCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
        }
        lnz++;
    }
    for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++)
    {
        // add L(i,kc) * U(kc,kc+1); keep union rows even when the term (or
        // the whole entry) is zero -- the forward strip needs to visit them
        int64_t i = L->i[p];
        if (i <= kc + 1) continue;              // diagonal / corner rows
        if (w[i] >= 0)
        {
            if (uk_pos >= 0)
            {
                mpz_addmul (Lp->x.mpz[w[i]], L->x.mpz[p], U->x.mpz[uk_pos]);
            }
        }
        else
        {
            w[i] = lnz;
            Lp->i[lnz] = i;
            if (uk_pos >= 0)
            {
                PCK (SLIP_mpz_mul (Lp->x.mpz[lnz], L->x.mpz[p],
                    U->x.mpz[uk_pos]));
            }
            else
            {
                PCK (SLIP_mpz_set_ui (Lp->x.mpz[lnz], 0));
            }
            lnz++;
        }
    }
    for (int64_t p = strip_start; p < lnz; p++)     // exact division
    {
        PCK (SLIP_mpz_divexact (Lp->x.mpz[p], Lp->x.mpz[p],
            rhos->x.mpz[kc]));
    }
    int64_t strip_end = lnz;
    for (int64_t p = strip_start; p < strip_end; p++)   // reset workspace
    {
        w[Lp->i[p]] = -1;
    }

    // column kc+1
    Lp->p[kc+1] = lnz;
    if (blk)
    {
        // block form: B'(kc,kc+1) = old L(kc+1,kc); "diagonal" B'(kc+1,kc+1)
        // = old rho[kc]; below the block, the old column kc verbatim
        Lp->i[lnz] = kc;
        PCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[lk_pos]));
        lnz++;
        Lp->i[lnz] = kc + 1;
        PCK (SLIP_mpz_set (Lp->x.mpz[lnz], rhos->x.mpz[kc]));
        lnz++;
        for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++)
        {
            if (L->i[p] < kc + 2) continue;
            Lp->i[lnz] = L->i[p];
            PCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
            lnz++;
        }
    }
    else
    {
        // diagonal rho[kc+1] unchanged; forward strip
        //   L'(i,kc+1) = ( rho'[kc]*L(i,kc) - L'(i,kc)*L(kc+1,kc) ) / rho[kc-1]
        for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++) // scatter old col kc
        {
            if (L->i[p] >= kc + 2) w[L->i[p]] = p;
        }
        Lp->i[lnz] = kc + 1;
        PCK (SLIP_mpz_set (Lp->x.mpz[lnz], rhos->x.mpz[kc+1]));
        lnz++;
        for (int64_t q = strip_start; q < strip_end; q++)
        {
            int64_t i = Lp->i[q];
            if (w[i] >= 0)
            {
                PCK (SLIP_mpz_mul (*t1, rp->x.mpz[kc], L->x.mpz[w[i]]));
            }
            else
            {
                PCK (SLIP_mpz_set_ui (*t1, 0));
            }
            if (lk_pos >= 0)
            {
                PCK (SLIP_mpz_submul (*t1, Lp->x.mpz[q], L->x.mpz[lk_pos]));
            }
            if (kc > 0)
            {
                PCK (SLIP_mpz_divexact (*t1, *t1, rhos->x.mpz[kc-1]));
            }
            PCK (SLIP_mpz_sgn (&sgn, *t1));
            if (sgn != 0)
            {
                Lp->i[lnz] = i;
                PCK (SLIP_mpz_set (Lp->x.mpz[lnz], *t1));
                lnz++;
            }
        }
        for (int64_t p = L->p[kc]; p < L->p[kc+1]; p++) // reset workspace
        {
            if (L->i[p] >= kc + 2) w[L->i[p]] = -1;
        }
    }

    for (int64_t j = kc + 2; j < n; j++)    // trailing cols: verbatim
    {
        Lp->p[j] = lnz;
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            Lp->i[lnz] = L->i[p];
            PCK (SLIP_mpz_set (Lp->x.mpz[lnz], L->x.mpz[p]));
            lnz++;
        }
    }
    Lp->p[n] = lnz;

    //--------------------------------------------------------------------------
    // build the new U
    //--------------------------------------------------------------------------

    PCK (SLIP_matrix_allocate (&Up, SLIP_CSC, SLIP_MPZ, n, n,
        U->p[n] + 2 * (n - kc) + 4, false, true, option));

    int64_t unz = 0;
    for (int64_t j = 0; j < kc; j++)        // cols < kc: verbatim
    {
        Up->p[j] = unz;
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            Up->i[unz] = U->i[p];
            PCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
            unz++;
        }
    }

    // column kc: rows < kc from the OLD column kc+1 (Step 5 swap); new
    // diagonal rho'[kc], or in block mode the below-diagonal block entry
    // B'(kc+1,kc) = old U(kc,kc+1) (the zero diagonal is not stored)
    Up->p[kc] = unz;
    for (int64_t p = U->p[kc+1]; p < U->p[kc+2]; p++)
    {
        int64_t i = U->i[p];
        if (i >= kc) continue;              // (kc,kc+1) and diagonal: corner
        Up->i[unz] = i;
        PCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
        unz++;
    }
    if (!blk)
    {
        Up->i[unz] = kc;
        PCK (SLIP_mpz_set (Up->x.mpz[unz], rp->x.mpz[kc]));
        unz++;
    }
    else
    {
        Up->i[unz] = kc + 1;
        PCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[uk_pos]));
        unz++;
    }

    // column kc+1: rows < kc from the OLD column kc (Step 5 swap); corner
    // U'(kc,kc+1) = old L(kc+1,kc) (= B'(kc,kc+1) in block mode); diagonal
    // rho[kc+1] unchanged, or old rho[kc] (= B'(kc+1,kc+1)) in block mode
    Up->p[kc+1] = unz;
    for (int64_t p = U->p[kc]; p < U->p[kc+1]; p++)
    {
        int64_t i = U->i[p];
        if (i >= kc) continue;              // old diagonal: replaced
        Up->i[unz] = i;
        PCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
        unz++;
    }
    if (lk_pos >= 0)
    {
        Up->i[unz] = kc;
        PCK (SLIP_mpz_set (Up->x.mpz[unz], L->x.mpz[lk_pos]));
        unz++;
    }
    Up->i[unz] = kc + 1;
    PCK (SLIP_mpz_set (Up->x.mpz[unz], rhos->x.mpz[blk ? kc : kc+1]));
    unz++;

    // columns >= kc+2: rows kc and kc+1 recomputed by strips 6c/6d, the
    // rest verbatim
    for (int64_t j = kc + 2; j < n; j++)
    {
        Up->p[j] = unz;

        // pass 1: locate U(kc,j) and U(kc+1,j) in the old column
        int64_t pk = -1, pk1 = -1;
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            if (U->i[p] == kc) pk = p;
            else if (U->i[p] == kc + 1) pk1 = p;
        }

        // 6c into t1, 6d into t2 (zero inputs handled via presence flags)
        bool have = (pk >= 0 || pk1 >= 0);
        if (have)
        {
            // U'(kc,j) = ( rho[kc-1]*U(kc+1,j) + L(kc+1,kc)*U(kc,j) ) / rho[kc]
            if (pk1 >= 0)
            {
                if (kc > 0)
                {
                    PCK (SLIP_mpz_mul (*t1, rhos->x.mpz[kc-1],
                        U->x.mpz[pk1]));
                }
                else
                {
                    PCK (SLIP_mpz_set (*t1, U->x.mpz[pk1]));
                }
            }
            else
            {
                PCK (SLIP_mpz_set_ui (*t1, 0));
            }
            if (pk >= 0 && lk_pos >= 0)
            {
                mpz_addmul (*t1, L->x.mpz[lk_pos], U->x.mpz[pk]);
            }
            PCK (SLIP_mpz_divexact (*t1, *t1, rhos->x.mpz[kc]));

            if (blk)
            {
                // block: U'(kc+1,j) = old U(kc,j), verbatim
                if (pk >= 0)
                {
                    PCK (SLIP_mpz_set (*t2, U->x.mpz[pk]));
                }
                else
                {
                    PCK (SLIP_mpz_set_ui (*t2, 0));
                }
            }
            else
            {
                // U'(kc+1,j) = ( rho'[kc]*U(kc,j) - U(kc,kc+1)*U'(kc,j) ) / rho[kc-1]
                if (pk >= 0)
                {
                    PCK (SLIP_mpz_mul (*t2, rp->x.mpz[kc], U->x.mpz[pk]));
                }
                else
                {
                    PCK (SLIP_mpz_set_ui (*t2, 0));
                }
                if (uk_pos >= 0)
                {
                    PCK (SLIP_mpz_submul (*t2, U->x.mpz[uk_pos], *t1));
                }
                if (kc > 0)
                {
                    PCK (SLIP_mpz_divexact (*t2, *t2, rhos->x.mpz[kc-1]));
                }
            }
        }

        // pass 2: emit the column
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            int64_t i = U->i[p];
            Up->i[unz] = i;
            if (i == kc)
            {
                PCK (SLIP_mpz_set (Up->x.mpz[unz], *t1));
            }
            else if (i == kc + 1)
            {
                PCK (SLIP_mpz_set (Up->x.mpz[unz], *t2));
            }
            else
            {
                PCK (SLIP_mpz_set (Up->x.mpz[unz], U->x.mpz[p]));
            }
            unz++;
        }
        if (have && pk < 0)                 // fill-in in row kc
        {
            PCK (SLIP_mpz_sgn (&sgn, *t1));
            if (sgn != 0)
            {
                Up->i[unz] = kc;
                PCK (SLIP_mpz_set (Up->x.mpz[unz], *t1));
                unz++;
            }
        }
        if (have && pk1 < 0)                // fill-in in row kc+1
        {
            PCK (SLIP_mpz_sgn (&sgn, *t2));
            if (sgn != 0)
            {
                Up->i[unz] = kc + 1;
                PCK (SLIP_mpz_set (Up->x.mpz[unz], *t2));
                unz++;
            }
        }
    }
    Up->p[n] = unz;

    // hand the updated factors to the caller
    *L_out = Lp;      Lp = NULL;
    *U_out = Up;      Up = NULL;
    *rhos_out = rp;   rp = NULL;

done:
    SLIP_matrix_free (&Lp, option);
    SLIP_matrix_free (&Up, option);
    SLIP_matrix_free (&rp, option);
    SLIP_matrix_free (&tmp, option);
    SLIP_FREE (w);
    #undef PCK
    return info;
}

//------------------------------------------------------------------------------
// distant diagonal push: an IN-PLACE chain of adjacent diagonal pushes
//------------------------------------------------------------------------------

/* The distant push moves the row/column at position k of PAQ to position m
 * (a symmetric cyclic rotation of the positions in between) as a chain of
 * |m - k| adjacent diagonal pushes.  Rebuilding the CSC factors on every
 * push would pay O(nnz(L) + nnz(U)) per push just copying unchanged
 * entries; here the factors are held so that one push touches ONLY the
 * affected rows and columns:
 *
 *   - L is stored as one resizable sparse vector per COLUMN and U as one
 *     per ROW.  A push at position j physically rebuilds L columns j, j+1
 *     and U rows j, j+1 -- the only vectors whose VALUES change -- and
 *     leaves every other vector untouched.
 *   - The rest of the push is pure relabeling: rows j and j+1 of L swap
 *     inside the leading columns, and columns j and j+1 of U swap inside
 *     the leading rows.  Entry indices are therefore kept as STORED labels,
 *     with overlays mapping stored labels to current positions and back --
 *     one for the rows of L (rpos_*) and one for the columns of U
 *     (cpos_*).  A diagonal push swaps the pair in BOTH overlays, a column
 *     push (dyn_colpush_step below) only in the column overlay, each in
 *     O(1) instead of visiting every leading vector; the trailing vectors
 *     never contain rows/columns j, j+1, so the swap is vacuous there.
 *
 * Per-push cost is O(nnz of L cols j, j+1 + nnz of U rows j, j+1) exact
 * arithmetic, independent of n and of the chain length.  The pivots rho[]
 * are indexed by current position and only rho[j] changes (the leading
 * (j+2)-minor is invariant under a symmetric interior swap), so no overlay
 * is needed for them.  Raw GMP calls are used throughout (they cannot fail
 * short of aborting on OOM), matching the existing raw-GMP usage here.
 */

typedef struct
{
    int64_t nz;         // entries in use
    int64_t cap;        // entries allocated; x[0..cap-1] are initialized
    int64_t *idx;       // stored labels (row of L / column of U)
    mpz_t *x;           // values
} dvec;

/* grow v so it can hold at least need entries (never shrinks) */
static SLIP_info dvec_reserve (dvec *v, int64_t need)
{
    if (need <= v->cap) return SLIP_OK;
    int64_t cap = (2 * v->cap > need) ? 2 * v->cap : need;
    int64_t *idx = (int64_t *) SLIP_malloc (cap * sizeof (int64_t));
    mpz_t *x = (mpz_t *) SLIP_malloc (cap * sizeof (mpz_t));
    if (idx == NULL || x == NULL)
    {
        SLIP_FREE (idx);
        SLIP_FREE (x);
        return SLIP_OUT_OF_MEMORY;
    }
    if (v->cap > 0)
    {
        memcpy (idx, v->idx, v->cap * sizeof (int64_t));
        memcpy (x, v->x, v->cap * sizeof (mpz_t));  // mpz_t moves by memcpy
    }
    for (int64_t p = v->cap; p < cap; p++) mpz_init (x[p]);
    SLIP_FREE (v->idx);
    SLIP_FREE (v->x);   // the moved mpz contents live on in x
    v->idx = idx;
    v->x = x;
    v->cap = cap;
    return SLIP_OK;
}

static void dvec_clear (dvec *v)
{
    for (int64_t p = 0; p < v->cap; p++) mpz_clear (v->x[p]);
    SLIP_FREE (v->idx);
    SLIP_FREE (v->x);
    v->nz = v->cap = 0;
}

/* O(1) exchange of two vectors' contents */
static void dvec_swap (dvec *a, dvec *b)
{
    dvec t = *a;
    *a = *b;
    *b = t;
}

/* append the entry (i, val) */
static SLIP_info dvec_push (dvec *v, int64_t i, const mpz_t val)
{
    SLIP_info info = dvec_reserve (v, v->nz + 1);
    if (info != SLIP_OK) return info;
    v->idx[v->nz] = i;
    mpz_set (v->x[v->nz], val);
    v->nz++;
    return SLIP_OK;
}

/* free an array of n dvecs */
static void dmat_free (dvec **M_handle, int64_t n)
{
    dvec *M = *M_handle;
    if (M == NULL) return;
    for (int64_t k = 0; k < n; k++) dvec_clear (&M[k]);
    SLIP_FREE (M);
    *M_handle = NULL;
}

/* the columns of a CSC MPZ matrix as dynamic vectors (idx = row) */
static SLIP_info csc_to_dvec_cols (dvec **M_out, const SLIP_matrix *A)
{
    int64_t n = A->n;
    dvec *M = (dvec *) SLIP_calloc (n, sizeof (dvec));
    if (M == NULL) return SLIP_OUT_OF_MEMORY;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = A->p[j]; p < A->p[j+1]; p++)
        {
            if (dvec_push (&M[j], A->i[p], A->x.mpz[p]) != SLIP_OK)
            {
                dmat_free (&M, n);
                return SLIP_OUT_OF_MEMORY;
            }
        }
    }
    *M_out = M;
    return SLIP_OK;
}

/* the rows of a CSC MPZ matrix as dynamic vectors (idx = column) */
static SLIP_info csc_to_dvec_rows (dvec **M_out, const SLIP_matrix *A)
{
    int64_t n = A->n;
    dvec *M = (dvec *) SLIP_calloc (n, sizeof (dvec));
    if (M == NULL) return SLIP_OUT_OF_MEMORY;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = A->p[j]; p < A->p[j+1]; p++)
        {
            if (dvec_push (&M[A->i[p]], j, A->x.mpz[p]) != SLIP_OK)
            {
                dmat_free (&M, n);
                return SLIP_OUT_OF_MEMORY;
            }
        }
    }
    *M_out = M;
    return SLIP_OK;
}

/* column lists back to CSC, mapping stored row labels through pos_cur;
 * the values are MOVED out of the lists (mpz_swap, no limb copies) */
static SLIP_info dvec_cols_to_csc (SLIP_matrix **A_out, dvec *M, int64_t n,
    const int64_t *pos_cur, const SLIP_options *option)
{
    SLIP_info info;
    int64_t nnz = 0;
    for (int64_t c = 0; c < n; c++) nnz += M[c].nz;
    SLIP_matrix *C = NULL;
    info = SLIP_matrix_allocate (&C, SLIP_CSC, SLIP_MPZ, n, n,
        nnz > 0 ? nnz : 1, false, true, option);
    if (info != SLIP_OK) return info;
    int64_t nz = 0;
    for (int64_t c = 0; c < n; c++)
    {
        C->p[c] = nz;
        for (int64_t p = 0; p < M[c].nz; p++)
        {
            C->i[nz] = pos_cur[M[c].idx[p]];
            mpz_swap (C->x.mpz[nz], M[c].x[p]);
            nz++;
        }
    }
    C->p[n] = nz;
    *A_out = C;
    return SLIP_OK;
}

/* row lists back to CSC, mapping stored column labels through pos_cur;
 * the values are MOVED out of the lists */
static SLIP_info dvec_rows_to_csc (SLIP_matrix **A_out, dvec *M, int64_t n,
    const int64_t *pos_cur, const SLIP_options *option)
{
    SLIP_info info;
    int64_t nnz = 0;
    for (int64_t r = 0; r < n; r++) nnz += M[r].nz;
    SLIP_matrix *C = NULL;
    info = SLIP_matrix_allocate (&C, SLIP_CSC, SLIP_MPZ, n, n,
        nnz > 0 ? nnz : 1, false, true, option);
    if (info != SLIP_OK) return info;
    int64_t *next = (int64_t *) SLIP_calloc (n, sizeof (int64_t));
    if (next == NULL)
    {
        SLIP_matrix_free (&C, option);
        return SLIP_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < n; r++)         // column counts
    {
        for (int64_t p = 0; p < M[r].nz; p++) next[pos_cur[M[r].idx[p]]]++;
    }
    C->p[0] = 0;
    for (int64_t c = 0; c < n; c++)         // column pointers
    {
        C->p[c+1] = C->p[c] + next[c];
        next[c] = C->p[c];                  // reuse as the next free slot
    }
    for (int64_t r = 0; r < n; r++)         // scatter (rows land sorted)
    {
        for (int64_t p = 0; p < M[r].nz; p++)
        {
            int64_t pos = next[pos_cur[M[r].idx[p]]]++;
            C->i[pos] = r;
            mpz_swap (C->x.mpz[pos], M[r].x[p]);
        }
    }
    SLIP_FREE (next);
    *A_out = C;
    return SLIP_OK;
}

/* the state threaded through the chain */
typedef struct
{
    int64_t n;
    dvec *Lc;               // the n columns of L
    dvec *Ur;               // the n rows of U
    mpz_t *rho;             // pivots by current position (owned by caller)
    int64_t *rpos_cur;      // L rows:    stored label -> current position
    int64_t *rpos_stored;   // L rows:    current position -> stored label
    int64_t *cpos_cur;      // U columns: stored label -> current position
    int64_t *cpos_stored;   // U columns: current position -> stored label
    int64_t *w;             // size-n scatter workspace, all -1 between pushes
    dvec s[4];              // scratch for the four rebuilt vectors
    mpz_t t[2];             // t[0] = rho', t[1] = strip scratch
} dyn_chain;

/* One adjacent diagonal push at current position j, in place: the same
 * mathematics as apdpu_sparse (closed-form corner, strips 6a-6d, 2x2 block
 * pivot fallback), restated on the dynamic storage.  Only L columns j, j+1
 * and U rows j, j+1 are rebuilt (into the scratch vectors, then swapped in
 * O(1)); the leading-vector swaps are the O(1) label-overlay update; strip
 * union zeros are dropped once the forward strip has consumed them, so no
 * explicit zeros accumulate over a chain.  Returns SLIP_SINGULAR if the new
 * pivot is zero and allow_block is false; with allow_block it engages the
 * block fallback and sets *used_block.
 */
static SLIP_info dyn_push_step (dyn_chain *C, int64_t j, bool allow_block,
    bool *used_block)
{
    SLIP_info info;
    int64_t rsj  = C->rpos_stored[j];       // stored label of current row j
    int64_t rsj1 = C->rpos_stored[j+1];     // ... and of current row j+1
    int64_t csj  = C->cpos_stored[j];       // stored label of current col j
    int64_t csj1 = C->cpos_stored[j+1];     // ... and of current col j+1
    dvec *Lj  = &C->Lc[j], *Lj1 = &C->Lc[j+1];
    dvec *Uj  = &C->Ur[j], *Uj1 = &C->Ur[j+1];
    dvec *nLj = &C->s[0], *nLj1 = &C->s[1];
    dvec *nUj = &C->s[2], *nUj1 = &C->s[3];
    int64_t *w = C->w;
    mpz_t *rho = C->rho;
    mpz_ptr rp = C->t[0], t1 = C->t[1];

    #define DPK(method) { info = (method); if (info != SLIP_OK) return info; }

    //--------------------------------------------------------------------------
    // locate the stored couplings and compute the new pivot
    //--------------------------------------------------------------------------

    int64_t pu = -1;                        // U(j,j+1), -1 if zero
    for (int64_t p = 0; p < Uj->nz; p++)
    {
        if (Uj->idx[p] == csj1) { pu = p; break; }
    }
    int64_t pl = -1;                        // L(j+1,j), -1 if zero
    for (int64_t p = 0; p < Lj->nz; p++)
    {
        if (Lj->idx[p] == rsj1) { pl = p; break; }
    }

    // rho' = ( rho[j-1]*rho[j+1] + L(j+1,j)*U(j,j+1) ) / rho[j]
    if (j > 0) mpz_mul (rp, rho[j-1], rho[j+1]);
    else       mpz_set (rp, rho[j+1]);
    if (pu >= 0 && pl >= 0) mpz_addmul (rp, Lj->x[pl], Uj->x[pu]);
    mpz_divexact (rp, rp, rho[j]);

    bool blk = (mpz_sgn (rp) == 0);         // 2x2 block pivot fallback
    if (blk && !allow_block) return SLIP_SINGULAR;
    if (blk && (pu < 0 || pl < 0)) return SLIP_SINGULAR;   // cannot happen:
        // the block determinant -L(j+1,j)*U(j,j+1) = rho[j-1]*rho[j+1] != 0
    if (used_block != NULL) *used_block = blk;

    //--------------------------------------------------------------------------
    // new L column j: diagonal rho' (omitted in block mode), corner
    // L'(j+1,j) = U(j,j+1), backtrack strip over the union pattern
    //--------------------------------------------------------------------------

    nLj->nz = 0;
    DPK (dvec_reserve (nLj, Lj->nz + Lj1->nz + 2));
    if (!blk)
    {
        nLj->idx[nLj->nz] = rsj1;           // current row j after the swap
        mpz_set (nLj->x[nLj->nz], rp);
        nLj->nz++;
    }
    if (pu >= 0)
    {
        nLj->idx[nLj->nz] = rsj;            // current row j+1 after the swap
        mpz_set (nLj->x[nLj->nz], Uj->x[pu]);
        nLj->nz++;
    }
    int64_t strip0 = nLj->nz;
    for (int64_t p = 0; p < Lj1->nz; p++)   // scatter rho[j-1] * L(i,j+1)
    {
        int64_t i = Lj1->idx[p];
        if (i == rsj1) continue;            // old diagonal: corner case
        w[i] = nLj->nz;
        nLj->idx[nLj->nz] = i;
        if (j > 0) mpz_mul (nLj->x[nLj->nz], rho[j-1], Lj1->x[p]);
        else       mpz_set (nLj->x[nLj->nz], Lj1->x[p]);
        nLj->nz++;
    }
    for (int64_t p = 0; p < Lj->nz; p++)    // add L(i,j) * U(j,j+1); keep
    {                                       // union rows even when zero
        int64_t i = Lj->idx[p];
        if (i == rsj || i == rsj1) continue;    // diagonal / corner rows
        if (w[i] >= 0)
        {
            if (pu >= 0) mpz_addmul (nLj->x[w[i]], Lj->x[p], Uj->x[pu]);
        }
        else
        {
            w[i] = nLj->nz;
            nLj->idx[nLj->nz] = i;
            if (pu >= 0) mpz_mul (nLj->x[nLj->nz], Lj->x[p], Uj->x[pu]);
            else         mpz_set_ui (nLj->x[nLj->nz], 0);
            nLj->nz++;
        }
    }
    for (int64_t p = strip0; p < nLj->nz; p++)  // exact division; reset w
    {
        mpz_divexact (nLj->x[p], nLj->x[p], rho[j]);
        w[nLj->idx[p]] = -1;
    }

    //--------------------------------------------------------------------------
    // new L column j+1: forward strip 6b, or the block-mode verbatim column
    //--------------------------------------------------------------------------

    nLj1->nz = 0;
    if (blk)
    {
        // B'(j,j+1) = old L(j+1,j); "diagonal" B'(j+1,j+1) = old rho[j];
        // below the block, the old column j verbatim
        DPK (dvec_reserve (nLj1, Lj->nz + 2));
        nLj1->idx[nLj1->nz] = rsj1;         // current row j
        mpz_set (nLj1->x[nLj1->nz], Lj->x[pl]);
        nLj1->nz++;
        nLj1->idx[nLj1->nz] = rsj;          // current row j+1
        mpz_set (nLj1->x[nLj1->nz], rho[j]);
        nLj1->nz++;
        for (int64_t p = 0; p < Lj->nz; p++)
        {
            int64_t i = Lj->idx[p];
            if (i == rsj || i == rsj1) continue;
            nLj1->idx[nLj1->nz] = i;
            mpz_set (nLj1->x[nLj1->nz], Lj->x[p]);
            nLj1->nz++;
        }
    }
    else
    {
        // L'(i,j+1) = ( rho'*L(i,j) - L'(i,j)*L(j+1,j) ) / rho[j-1]
        DPK (dvec_reserve (nLj1, (nLj->nz - strip0) + 1));
        nLj1->idx[nLj1->nz] = rsj;          // diagonal rho[j+1], unchanged
        mpz_set (nLj1->x[nLj1->nz], rho[j+1]);
        nLj1->nz++;
        for (int64_t p = 0; p < Lj->nz; p++)    // scatter the old column j
        {
            int64_t i = Lj->idx[p];
            if (i != rsj && i != rsj1) w[i] = p;
        }
        for (int64_t p = strip0; p < nLj->nz; p++)
        {
            int64_t i = nLj->idx[p];
            if (w[i] >= 0) mpz_mul (t1, rp, Lj->x[w[i]]);
            else           mpz_set_ui (t1, 0);
            if (pl >= 0) mpz_submul (t1, nLj->x[p], Lj->x[pl]);
            if (j > 0) mpz_divexact (t1, t1, rho[j-1]);
            if (mpz_sgn (t1) != 0)
            {
                nLj1->idx[nLj1->nz] = i;
                mpz_set (nLj1->x[nLj1->nz], t1);
                nLj1->nz++;
            }
        }
        for (int64_t p = 0; p < Lj->nz; p++)    // reset the workspace
        {
            int64_t i = Lj->idx[p];
            if (i != rsj && i != rsj1) w[i] = -1;
        }
    }

    // the forward strip has consumed the union: drop the explicit zeros
    int64_t q = 0;
    for (int64_t p = 0; p < nLj->nz; p++)
    {
        if (mpz_sgn (nLj->x[p]) != 0)
        {
            if (q != p)
            {
                nLj->idx[q] = nLj->idx[p];
                mpz_swap (nLj->x[q], nLj->x[p]);
            }
            q++;
        }
    }
    nLj->nz = q;

    //--------------------------------------------------------------------------
    // new U row j: diagonal rho' (omitted in block mode), corner
    // U'(j,j+1) = L(j+1,j), backtrack strip 6c over the union pattern
    //--------------------------------------------------------------------------

    nUj->nz = 0;
    DPK (dvec_reserve (nUj, Uj->nz + Uj1->nz + 2));
    if (!blk)
    {
        nUj->idx[nUj->nz] = csj1;           // current column j after the swap
        mpz_set (nUj->x[nUj->nz], rp);
        nUj->nz++;
    }
    if (pl >= 0)
    {
        nUj->idx[nUj->nz] = csj;            // current column j+1 (corner)
        mpz_set (nUj->x[nUj->nz], Lj->x[pl]);
        nUj->nz++;
    }
    int64_t ustrip0 = nUj->nz;
    for (int64_t p = 0; p < Uj1->nz; p++)   // scatter rho[j-1] * U(j+1,c)
    {
        int64_t c = Uj1->idx[p];
        if (c == csj1) continue;            // old diagonal: corner case
        w[c] = nUj->nz;
        nUj->idx[nUj->nz] = c;
        if (j > 0) mpz_mul (nUj->x[nUj->nz], rho[j-1], Uj1->x[p]);
        else       mpz_set (nUj->x[nUj->nz], Uj1->x[p]);
        nUj->nz++;
    }
    for (int64_t p = 0; p < Uj->nz; p++)    // add L(j+1,j) * U(j,c); keep
    {                                       // union columns even when zero
        int64_t c = Uj->idx[p];
        if (c == csj || c == csj1) continue;    // diagonal / corner columns
        if (w[c] >= 0)
        {
            if (pl >= 0) mpz_addmul (nUj->x[w[c]], Lj->x[pl], Uj->x[p]);
        }
        else
        {
            w[c] = nUj->nz;
            nUj->idx[nUj->nz] = c;
            if (pl >= 0) mpz_mul (nUj->x[nUj->nz], Lj->x[pl], Uj->x[p]);
            else         mpz_set_ui (nUj->x[nUj->nz], 0);
            nUj->nz++;
        }
    }
    for (int64_t p = ustrip0; p < nUj->nz; p++) // exact division; reset w
    {
        mpz_divexact (nUj->x[p], nUj->x[p], rho[j]);
        w[nUj->idx[p]] = -1;
    }

    //--------------------------------------------------------------------------
    // new U row j+1: forward strip 6d, or the block-mode verbatim row
    //--------------------------------------------------------------------------

    nUj1->nz = 0;
    if (blk)
    {
        // B'(j+1,j) = old U(j,j+1); "diagonal" B'(j+1,j+1) = old rho[j];
        // beyond the block, the old row j verbatim
        DPK (dvec_reserve (nUj1, Uj->nz + 2));
        nUj1->idx[nUj1->nz] = csj1;         // current column j
        mpz_set (nUj1->x[nUj1->nz], Uj->x[pu]);
        nUj1->nz++;
        nUj1->idx[nUj1->nz] = csj;          // current column j+1
        mpz_set (nUj1->x[nUj1->nz], rho[j]);
        nUj1->nz++;
        for (int64_t p = 0; p < Uj->nz; p++)
        {
            int64_t c = Uj->idx[p];
            if (c == csj || c == csj1) continue;
            nUj1->idx[nUj1->nz] = c;
            mpz_set (nUj1->x[nUj1->nz], Uj->x[p]);
            nUj1->nz++;
        }
    }
    else
    {
        // U'(j+1,c) = ( rho'*U(j,c) - U(j,j+1)*U'(j,c) ) / rho[j-1]
        DPK (dvec_reserve (nUj1, (nUj->nz - ustrip0) + 1));
        nUj1->idx[nUj1->nz] = csj;          // diagonal rho[j+1], unchanged
        mpz_set (nUj1->x[nUj1->nz], rho[j+1]);
        nUj1->nz++;
        for (int64_t p = 0; p < Uj->nz; p++)    // scatter the old row j
        {
            int64_t c = Uj->idx[p];
            if (c != csj && c != csj1) w[c] = p;
        }
        for (int64_t p = ustrip0; p < nUj->nz; p++)
        {
            int64_t c = nUj->idx[p];
            if (w[c] >= 0) mpz_mul (t1, rp, Uj->x[w[c]]);
            else           mpz_set_ui (t1, 0);
            if (pu >= 0) mpz_submul (t1, Uj->x[pu], nUj->x[p]);
            if (j > 0) mpz_divexact (t1, t1, rho[j-1]);
            if (mpz_sgn (t1) != 0)
            {
                nUj1->idx[nUj1->nz] = c;
                mpz_set (nUj1->x[nUj1->nz], t1);
                nUj1->nz++;
            }
        }
        for (int64_t p = 0; p < Uj->nz; p++)    // reset the workspace
        {
            int64_t c = Uj->idx[p];
            if (c != csj && c != csj1) w[c] = -1;
        }
    }

    q = 0;                                  // drop the union zeros in row j
    for (int64_t p = 0; p < nUj->nz; p++)
    {
        if (mpz_sgn (nUj->x[p]) != 0)
        {
            if (q != p)
            {
                nUj->idx[q] = nUj->idx[p];
                mpz_swap (nUj->x[q], nUj->x[p]);
            }
            q++;
        }
    }
    nUj->nz = q;

    //--------------------------------------------------------------------------
    // commit: swap in the rebuilt vectors, update the pivot and the overlay
    //--------------------------------------------------------------------------

    dvec_swap (Lj, nLj);
    dvec_swap (Lj1, nLj1);
    dvec_swap (Uj, nUj);
    dvec_swap (Uj1, nUj1);
    mpz_swap (rho[j], rp);                  // rho' (0 in block mode)
    C->rpos_stored[j] = rsj1;               // the O(1) leading-vector swaps:
    C->rpos_stored[j+1] = rsj;              // the symmetric push exchanges
    C->rpos_cur[rsj] = j + 1;               // both the row pair and the
    C->rpos_cur[rsj1] = j;                  // column pair
    C->cpos_stored[j] = csj1;
    C->cpos_stored[j+1] = csj;
    C->cpos_cur[csj] = j + 1;
    C->cpos_cur[csj1] = j;

    #undef DPK
    return SLIP_OK;
}

/* One adjacent COLUMN push at current position j, in place: the matrix
 * columns at positions j and j+1 are exchanged and the rows stay put --
 * apcpu_sparse restated on the dynamic storage.  Sets *did = false and
 * changes nothing if the upper support U(j,j+1) is zero (the swapped matrix
 * then has no REF LU with this row order); the caller falls back to the
 * diagonal push, whose pivot rho[j-1]*rho[j+1]/rho[j] is then guaranteed
 * nonzero (Sylvester with a vanishing cross term).  Together the two steps
 * make a rightward chain that CANNOT be blocked on a nonsingular matrix.
 *
 * DEFERRED TRANSPOSITION SIGNS.  A column swap flips the sign of every
 * minor whose column set contains both swapped columns: L columns >= j+1,
 * U rows >= j+2, and rho[j+1..] -- the whole trailing region, which an
 * in-place chain cannot afford to visit.  The flips are therefore left
 * PENDING in the stored values, and the chain driver settles them once at
 * the end (see distant_push_inplace).  The bookkeeping is one global
 * parity: with e = (-1)^(column pushes performed so far), every LIVE
 * vector (L columns >= j, U rows >= j, pivots rho[j-1..]) stores e times
 * its true value, and every IPGE identity -- a degree-2 product over a
 * degree-1 divisor -- maps e-scaled inputs to e-scaled outputs, so the
 * arithmetic never sees the pending signs.  A column push advances the
 * parity to e' = -e:
 *
 *   - the trailing region is left untouched: its true values flip, so its
 *     stored values are at parity e' automatically;
 *   - the two rebuilt vectors are written at parity e':
 *       stored rho'[j] = -stored U(j,j+1)          (true rho'[j] = U(j,j+1))
 *       stored L'(i,j) = -( rho[j-1]*L(i,j+1) + L(i,j)*U(j,j+1) ) / rho[j]
 *     and the U(j+1,:) strip formulas below, evaluated on rho'[j] and
 *     L'(j+1,j) (already at e') and old row-j/j+1 values (at e), land at
 *     parity e' with no further sign work;
 *   - U row j and everything at positions < j are FROZEN at the parity
 *     they held when the chain passed them, recorded by the driver.
 *
 * The rebuilt vectors:
 *
 *   new L column j (union of the old columns j, j+1; rows keep their
 *   labels -- no row swap), diagonal rho'[j], zeros dropped (no dependent
 *   forward strip here, unlike the diagonal push);
 *
 *   new U row j+1: diagonal rho[j+1] (stored value unchanged: the true
 *   pivot flips and so does the parity), and for c >= j+2 the apcpu strip
 *      xint       = ( rho[j-1]*U(j+1,c) + L(j+1,j)*U(j,c) ) / rho[j]
 *      U'(j+1,c)  = ( rho'[j]*xint - L'(j+1,j)*U(j,c) ) / rho[j-1]
 *   over the union of the old rows j, j+1.
 *
 * U row j needs NO edits: swapping the pair in the COLUMN overlay retargets
 * its old (j,j+1) entry to position j (the new diagonal, stored value
 * unchanged at the frozen parity) and its old diagonal to position j+1
 * (the corner U'(j,j+1) = old rho[j]), and moves the leading rows' entries
 * U(i,j) <-> U(i,j+1), i < j, in the same O(1) swap.  L's row overlay is
 * untouched.  Only rho[j] changes among the pivots.
 */
static SLIP_info dyn_colpush_step (dyn_chain *C, int64_t j, bool *did)
{
    SLIP_info info;
    int64_t rsj  = C->rpos_stored[j];       // row labels: unchanged here
    int64_t rsj1 = C->rpos_stored[j+1];
    int64_t csj  = C->cpos_stored[j];       // column labels: these swap
    int64_t csj1 = C->cpos_stored[j+1];
    dvec *Lj  = &C->Lc[j], *Lj1 = &C->Lc[j+1];
    dvec *Uj  = &C->Ur[j], *Uj1 = &C->Ur[j+1];
    dvec *nLj = &C->s[0], *nUj1 = &C->s[1], *xv = &C->s[2];
    int64_t *w = C->w;
    mpz_t *rho = C->rho;
    mpz_ptr rp = C->t[0], t1 = C->t[1];

    #define CPK(method) { info = (method); if (info != SLIP_OK) return info; }

    //--------------------------------------------------------------------------
    // feasibility: the new pivot is the upper support U(j,j+1)
    //--------------------------------------------------------------------------

    int64_t pu = -1;
    for (int64_t p = 0; p < Uj->nz; p++)
    {
        if (Uj->idx[p] == csj1) { pu = p; break; }
    }
    if (pu < 0 || mpz_sgn (Uj->x[pu]) == 0)
    {
        *did = false;                       // no column push at this j
        return SLIP_OK;
    }
    *did = true;

    int64_t pl = -1;                        // old L(j+1,j), -1 if zero
    for (int64_t p = 0; p < Lj->nz; p++)
    {
        if (Lj->idx[p] == rsj1) { pl = p; break; }
    }

    mpz_neg (rp, Uj->x[pu]);                // stored rho'[j], parity e'

    //--------------------------------------------------------------------------
    // new L column j over the union of the old columns j, j+1
    //--------------------------------------------------------------------------

    nLj->nz = 0;
    CPK (dvec_reserve (nLj, Lj->nz + Lj1->nz + 1));
    nLj->idx[nLj->nz] = rsj;                // diagonal = rho'[j]
    mpz_set (nLj->x[nLj->nz], rp);
    nLj->nz++;
    int64_t strip0 = nLj->nz;
    for (int64_t p = 0; p < Lj1->nz; p++)   // scatter rho[j-1] * L(i,j+1);
    {                                       // includes i = j+1, whose stored
        int64_t i = Lj1->idx[p];            // diagonal rho[j+1] belongs in
        w[i] = nLj->nz;                     // the formula
        nLj->idx[nLj->nz] = i;
        if (j > 0) mpz_mul (nLj->x[nLj->nz], rho[j-1], Lj1->x[p]);
        else       mpz_set (nLj->x[nLj->nz], Lj1->x[p]);
        nLj->nz++;
    }
    for (int64_t p = 0; p < Lj->nz; p++)    // add L(i,j) * U(j,j+1)
    {
        int64_t i = Lj->idx[p];
        if (i == rsj) continue;             // old diagonal not in the formula
        if (w[i] >= 0)
        {
            mpz_addmul (nLj->x[w[i]], Lj->x[p], Uj->x[pu]);
        }
        else
        {
            w[i] = nLj->nz;
            nLj->idx[nLj->nz] = i;
            mpz_mul (nLj->x[nLj->nz], Lj->x[p], Uj->x[pu]);
            nLj->nz++;
        }
    }
    for (int64_t p = strip0; p < nLj->nz; p++)  // exact division, parity flip
    {
        mpz_divexact (nLj->x[p], nLj->x[p], rho[j]);
        mpz_neg (nLj->x[p], nLj->x[p]);
        w[nLj->idx[p]] = -1;
    }
    // L'(j+1,j), needed by the strip below; row j+1 is always in the union
    // (it holds the old column j+1's diagonal), though cancellation can
    // leave a stored zero -- keep the union intact until the strip is done
    int64_t pln = -1;
    for (int64_t p = strip0; p < nLj->nz; p++)
    {
        if (nLj->idx[p] == rsj1) { pln = p; break; }
    }

    //--------------------------------------------------------------------------
    // new U row j+1: diagonal, then the strip over the union of rows j, j+1
    //--------------------------------------------------------------------------

    // xint over the union in columns >= j+2 (union zeros kept: the second
    // strip term can be nonzero where xint cancels to zero)
    xv->nz = 0;
    CPK (dvec_reserve (xv, Uj->nz + Uj1->nz));
    for (int64_t p = 0; p < Uj1->nz; p++)   // scatter rho[j-1] * U(j+1,c)
    {
        int64_t c = Uj1->idx[p];
        if (c == csj1) continue;            // old diagonal handled separately
        w[c] = xv->nz;
        xv->idx[xv->nz] = c;
        if (j > 0) mpz_mul (xv->x[xv->nz], rho[j-1], Uj1->x[p]);
        else       mpz_set (xv->x[xv->nz], Uj1->x[p]);
        xv->nz++;
    }
    for (int64_t p = 0; p < Uj->nz; p++)    // add L(j+1,j) * U(j,c)
    {
        int64_t c = Uj->idx[p];
        if (c == csj || c == csj1) continue;    // diagonal / new-pivot entries
        if (w[c] >= 0)
        {
            if (pl >= 0) mpz_addmul (xv->x[w[c]], Lj->x[pl], Uj->x[p]);
        }
        else
        {
            w[c] = xv->nz;
            xv->idx[xv->nz] = c;
            if (pl >= 0) mpz_mul (xv->x[xv->nz], Lj->x[pl], Uj->x[p]);
            else         mpz_set_ui (xv->x[xv->nz], 0);
            xv->nz++;
        }
    }
    for (int64_t p = 0; p < xv->nz; p++)    // exact division; reset w
    {
        mpz_divexact (xv->x[p], xv->x[p], rho[j]);
        w[xv->idx[p]] = -1;
    }

    nUj1->nz = 0;
    CPK (dvec_reserve (nUj1, xv->nz + 1));
    nUj1->idx[nUj1->nz] = csj;              // diagonal label AFTER the swap
    mpz_set (nUj1->x[nUj1->nz], rho[j+1]);  // stored value unchanged
    nUj1->nz++;
    for (int64_t p = 0; p < Uj->nz; p++)    // scatter the old row j
    {
        int64_t c = Uj->idx[p];
        if (c != csj && c != csj1) w[c] = p;
    }
    for (int64_t p = 0; p < xv->nz; p++)
    {
        int64_t c = xv->idx[p];
        mpz_mul (t1, rp, xv->x[p]);
        if (w[c] >= 0 && pln >= 0)
        {
            mpz_submul (t1, nLj->x[pln], Uj->x[w[c]]);
        }
        if (j > 0) mpz_divexact (t1, t1, rho[j-1]);
        if (mpz_sgn (t1) != 0)
        {
            nUj1->idx[nUj1->nz] = c;
            mpz_set (nUj1->x[nUj1->nz], t1);
            nUj1->nz++;
        }
    }
    for (int64_t p = 0; p < Uj->nz; p++)    // reset the workspace
    {
        int64_t c = Uj->idx[p];
        if (c != csj && c != csj1) w[c] = -1;
    }

    // the strip has consumed the union: drop L column j's explicit zeros
    int64_t q = 0;
    for (int64_t p = 0; p < nLj->nz; p++)
    {
        if (mpz_sgn (nLj->x[p]) != 0)
        {
            if (q != p)
            {
                nLj->idx[q] = nLj->idx[p];
                mpz_swap (nLj->x[q], nLj->x[p]);
            }
            q++;
        }
    }
    nLj->nz = q;

    //--------------------------------------------------------------------------
    // commit: swap in the rebuilt vectors, the pivot, and the column labels
    //--------------------------------------------------------------------------

    dvec_swap (Lj, nLj);
    dvec_swap (Uj1, nUj1);
    mpz_swap (rho[j], rp);
    C->cpos_stored[j] = csj1;               // O(1) column-side relabel; the
    C->cpos_stored[j+1] = csj;              // row overlay is untouched
    C->cpos_cur[csj] = j + 1;
    C->cpos_cur[csj1] = j;

    #undef CPK
    return SLIP_OK;
}

/* Distant push k -> m on the factors of PAQ, in place: convert the CSC
 * factors to the dynamic form once, run |m - k| O(local) pushes, and
 * convert back once.  Every intermediate state is the exact REF
 * factorization of a genuinely permuted matrix, so integrality holds
 * throughout, and PAQ itself is never consulted.
 *
 * With mixed = false the chain is the pure DIAGONAL push (the symmetric
 * cyclic rotation: rows and columns move together), semantics matching the
 * adjacent apdpu_sparse chained: a zero pivot on the LAST push engages the
 * 2x2 block pivot fallback and sets *used_block (rhos'[.] = 0 is the block
 * marker); a zero pivot in the INTERIOR aborts with SLIP_SINGULAR and
 * *blocked_at (pushing through a block would divide by the zero pivot in
 * the scalar Sylvester identities), and the caller should refactorize.
 *
 * With mixed = true (rightward chains only, k < m) each step performs the
 * DIAGONAL push when its pivot is nonzero and the COLUMN push otherwise
 * (PERM_COLPUSH_FIRST flips the preference).  The two pivots are coupled
 * by the Sylvester identity: the diagonal-push pivot is
 * ( rho[j-1]*rho[j+1] + L(j+1,j)*U(j,j+1) ) / rho[j] and the column-push
 * pivot is the upper support U(j,j+1), so if either vanishes the other is
 * nonzero (a vanishing upper support kills the cross term and leaves
 * rho[j-1]*rho[j+1]/rho[j] != 0; a vanishing diagonal pivot forces the
 * cross term, hence U(j,j+1), nonzero): ONE of the two pushes is always
 * feasible, the chain never hits a zero pivot, and the 2x2 block fallback
 * is never needed (used_block may be NULL) -- the theorem that motivates
 * the mixed chain.  Each column push defers its
 * trailing transposition sign flips (see dyn_colpush_step); they are
 * settled here in one O(nnz) pass before the CSC conversion: with g(p) =
 * (number of column pushes at chain positions <= p) mod 2, L column c and
 * rho[c] are negated iff g(c) is odd, and U row r iff g(r-1) is odd (each
 * vector was frozen at the parity the chain held when it passed).  The
 * emitted factors are the exact plain REF factors of R * PAQ * sigma,
 * where sigma is the same column cycle as the symmetric chain but R is
 * only the sub-permutation contributed by the diagonal-push steps; the row
 * map actually applied (old position -> new position) is returned in
 * rowmap_out[0..n-1] (pass NULL if not wanted), and *ncolpush_out gets the
 * number of column pushes taken (NULL ok).
 */
static SLIP_info distant_push_inplace (SLIP_matrix **L_out,
    SLIP_matrix **U_out, SLIP_matrix **rhos_out, SLIP_matrix *L,
    SLIP_matrix *U, SLIP_matrix *rhos, int64_t k, int64_t m, bool mixed,
    int64_t *rowmap_out, int64_t *ncolpush_out,
    bool *used_block, int64_t *blocked_at, const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *rp = NULL, *Lout = NULL, *Uout = NULL;
    char *ctype = NULL;     // mixed: 1 where the step was a column push
    dyn_chain C;
    memset (&C, 0, sizeof (C));
    mpz_init (C.t[0]);
    mpz_init (C.t[1]);

    if (used_block != NULL) *used_block = false;
    if (blocked_at != NULL) *blocked_at = -1;
    if (ncolpush_out != NULL) *ncolpush_out = 0;
    if (k < 0 || k >= n || m < 0 || m >= n || k == m || (mixed && k > m))
    {
        info = SLIP_INCORRECT_INPUT;
        goto done;
    }

    #define ICK(method) { info = (method); if (info != SLIP_OK) goto done; }

    C.n = n;
    ICK (csc_to_dvec_cols (&C.Lc, L));
    ICK (csc_to_dvec_rows (&C.Ur, U));
    ICK (SLIP_matrix_copy (&rp, SLIP_DENSE, SLIP_MPZ, rhos, option));
    C.rho = rp->x.mpz;
    C.rpos_cur    = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    C.rpos_stored = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    C.cpos_cur    = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    C.cpos_stored = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    C.w           = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    ctype         = (char *)    SLIP_calloc (n, sizeof (char));
    if (C.rpos_cur == NULL || C.rpos_stored == NULL ||
        C.cpos_cur == NULL || C.cpos_stored == NULL ||
        C.w == NULL || ctype == NULL)
    {
        info = SLIP_OUT_OF_MEMORY;
        goto done;
    }
    for (int64_t i = 0; i < n; i++)
    {
        C.rpos_cur[i] = i;      // stored labels start out as the positions
        C.rpos_stored[i] = i;
        C.cpos_cur[i] = i;
        C.cpos_stored[i] = i;
        C.w[i] = -1;
    }

    int64_t step   = (k < m) ? 1 : -1;
    int64_t nsteps = (k < m) ? (m - k) : (k - m);
    int64_t j      = (k < m) ? k : k - 1;   // first adjacent push position
    int64_t ncol   = 0;

    for (int64_t s = 0; s < nsteps; s++)
    {
        bool last = (s == nsteps - 1);
        bool blk = false;
        if (mixed)
        {
#ifndef PERM_COLPUSH_FIRST
            // Prefer the diagonal push: it is exactly the pure chain's
            // step, so the fill matches the symmetric chain wherever that
            // chain survives (and measures ~34% less total fill on the LP
            // test bases than preferring column pushes).  When its pivot
            // vanishes -- the old hard failure -- the identity
            // rho[j-1]*rho[j+1] + L(j+1,j)*U(j,j+1) = 0 forces a nonzero
            // upper support, so the column push below cannot decline.
            // Define PERM_COLPUSH_FIRST to flip the preference.
            info = dyn_push_step (&C, j, false, &blk);
            if (info != SLIP_SINGULAR)
            {
                ICK (info);
                j += step;
                continue;
            }
#endif
            bool did = false;
            ICK (dyn_colpush_step (&C, j, &did));
            if (did)
            {
                ctype[j] = 1;
                ncol++;
                j += step;
                continue;
            }
            // No column push at this j: with PERM_COLPUSH_FIRST the upper
            // support is zero and the diagonal push below cannot fail;
            // otherwise this is unreachable (the diagonal pivot vanished,
            // so the upper support was nonzero) and the dyn_push_step
            // below re-fails and reports blocked_at.
        }
        info = dyn_push_step (&C, j, !mixed && last && used_block != NULL,
            &blk);
        if (info == SLIP_SINGULAR)
        {
            if (blocked_at != NULL) *blocked_at = j;
            goto done;
        }
        ICK (info);
        if (last && used_block != NULL) *used_block = blk;
        j += step;
    }

    // settle the deferred transposition signs of the mixed chain's column
    // pushes: prefix parity over the chain positions, then one negation
    // pass (mpz_neg in place is O(1) per entry -- no limb work)
    if (ncol > 0)
    {
        int par = 0;
        for (int64_t c = 0; c < n; c++)
        {
            // par enters iteration c as g(c-1), leaves it as g(c)
            if (par)                        // U row c froze at parity g(c-1)
            {
                dvec *Ur = &C.Ur[c];
                for (int64_t p = 0; p < Ur->nz; p++)
                {
                    mpz_neg (Ur->x[p], Ur->x[p]);
                }
            }
            par ^= (int) ctype[c];
            if (par)                        // L col c and rho[c] at g(c)
            {
                dvec *Lc = &C.Lc[c];
                for (int64_t p = 0; p < Lc->nz; p++)
                {
                    mpz_neg (Lc->x[p], Lc->x[p]);
                }
                mpz_neg (C.rho[c], C.rho[c]);
            }
        }
    }

    ICK (dvec_cols_to_csc (&Lout, C.Lc, n, C.rpos_cur, option));
    ICK (dvec_rows_to_csc (&Uout, C.Ur, n, C.cpos_cur, option));
    if (rowmap_out != NULL)
    {
        memcpy (rowmap_out, C.rpos_cur, n * sizeof (int64_t));
    }
    if (ncolpush_out != NULL) *ncolpush_out = ncol;
    *L_out = Lout;      Lout = NULL;
    *U_out = Uout;      Uout = NULL;
    *rhos_out = rp;     rp = NULL;

done:
    mpz_clear (C.t[0]);
    mpz_clear (C.t[1]);
    for (int64_t i = 0; i < 4; i++) dvec_clear (&C.s[i]);
    dmat_free (&C.Lc, n);
    dmat_free (&C.Ur, n);
    SLIP_FREE (C.rpos_cur);
    SLIP_FREE (C.rpos_stored);
    SLIP_FREE (C.cpos_cur);
    SLIP_FREE (C.cpos_stored);
    SLIP_FREE (C.w);
    SLIP_FREE (ctype);
    SLIP_matrix_free (&rp, option);
    SLIP_matrix_free (&Lout, option);
    SLIP_matrix_free (&Uout, option);
    #undef ICK
    return info;
}

//------------------------------------------------------------------------------
// sparse PAQ' construction
//------------------------------------------------------------------------------

/* Fill sigma (and its inverse) with the permutation of positions that a
 * distant push k -> m applies to PAQ: new position j holds old position
 * sigma[j], so sigma[m] = k and the positions between k and m shift by one
 * place toward k; everything outside the window is fixed.  With m = k+1
 * this is the adjacent transposition of the single-swap workflow. */
static void cycle_sigma (int64_t *sigma, int64_t *sigma_inv, int64_t n,
    int64_t k, int64_t m)
{
    for (int64_t j = 0; j < n; j++) sigma[j] = j;
    if (k < m)
    {
        for (int64_t j = k; j < m; j++) sigma[j] = j + 1;
        sigma[m] = k;
    }
    else if (k > m)
    {
        for (int64_t j = m + 1; j <= k; j++) sigma[j] = j - 1;
        sigma[m] = k;
    }
    for (int64_t j = 0; j < n; j++) sigma_inv[sigma[j]] = j;
}

/* Build PAQ' -- PAQ with its columns permuted by sigma and its rows by an
 * optional row map:  PAQ'[i][j] = PAQ[rho[i]][sigma[j]] where rowmap is the
 * inverse of rho (rowmap[old position] = new position; NULL for identity)
 * -- directly in CSC form from A, pinv, and q, without a dense
 * intermediate:  column c of PAQ' is column q[sigma[c]] of A with row
 * indices mapped through pinv (composed with rowmap when given).  For the
 * symmetric permutation of the pure diagonal-push chain pass
 * rowmap = sigma_inv; a mixed column/diagonal-push chain passes the row
 * map it actually applied (see distant_push_inplace).
 */
static SLIP_info build_paq_csc (SLIP_matrix **PAQ_handle, const SLIP_matrix *A,
    const int64_t *pinv, const int64_t *q, const int64_t *sigma,
    const int64_t *rowmap, const SLIP_options *option)
{
    SLIP_info info;
    int64_t n = A->n, anz = A->p[A->n];
    SLIP_matrix *C = NULL;

    info = SLIP_matrix_allocate (&C, SLIP_CSC, SLIP_MPZ, n, n,
        anz > 0 ? anz : 1, false, true, option);
    if (info != SLIP_OK) return info;

    int64_t nz = 0;
    for (int64_t c = 0; c < n; c++)
    {
        int64_t j = q[sigma[c]];
        C->p[c] = nz;
        for (int64_t p = A->p[j]; p < A->p[j+1]; p++)
        {
            int64_t r = pinv[A->i[p]];
            if (rowmap != NULL) r = rowmap[r];
            C->i[nz] = r;
            info = SLIP_mpz_set (C->x.mpz[nz], A->x.mpz[p]);
            if (info != SLIP_OK) { SLIP_matrix_free (&C, option); return info; }
            nz++;
        }
    }
    C->p[n] = nz;

    *PAQ_handle = C;
    return SLIP_OK;
}

//------------------------------------------------------------------------------
// benchmark: distant diagonal push vs. full refactorization
//------------------------------------------------------------------------------

/* Benchmark one distant diagonal push k -> m (0-based) against a full
 * refactorization of the permuted matrix, and print one table row.
 *
 * The update side chains |m - k| adjacent diagonal pushes on the factors of
 * PAQ, in place (distant_push_inplace: each push touches only the two
 * affected columns of L and rows of U).  The refactorization side does what
 * a solver
 * without the update would do: build PAQ' and run the full SLIP pipeline on
 * it (fresh COLAMD analysis + factorization, 'option').  Correctness is
 * checked two ways, neither counted in the timing columns:
 *
 *   - determinant: |rhos'[n-1]| from the update must equal |rhos[n-1]| from
 *     the refactorization (both are +-det(PAQ'); available in every case);
 *   - exact factors: a second, fixed-order refactorization of PAQ'
 *     ('option2': no ordering, diagonal pivoting) is compared entrywise with
 *     the updated factors -- valid whenever it keeps P = identity (factor
 *     uniqueness).  When the chain ends in a 2x2 block pivot the fixed-order
 *     refactorization is forced to row-pivot, leaving the determinant test.
 *
 * A zero pivot in the interior of the chain aborts the update; the row then
 * reports the failing position and the refactorization stands as the
 * fallback (its time is still shown -- that is what the fallback costs).
 */
static SLIP_info bench_push (SLIP_matrix *A, SLIP_matrix *L1, SLIP_matrix *U1,
    SLIP_matrix *rhos1, const int64_t *pinv1, const int64_t *q1,
    int64_t k, int64_t m, SLIP_options *option, SLIP_options *option2)
{
    SLIP_info info = SLIP_OK;
    int64_t n = A->n;
    SLIP_matrix *L_s = NULL, *U_s = NULL, *rhos_s = NULL;   // updated factors
    SLIP_matrix *Ap = NULL;                                 // PAQ' (CSC)
    SLIP_LU_analysis *Sr = NULL, *Sv = NULL;
    SLIP_matrix *Lr = NULL, *Ur = NULL, *rr = NULL;         // refactorization
    SLIP_matrix *Lv = NULL, *Uv = NULL, *rv = NULL;         // fixed-order verify
    int64_t *pr = NULL, *pv = NULL;
    int64_t *sigma = NULL, *sigma_inv = NULL;
    clock_t tic, toc;

    #define BCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    sigma     = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    sigma_inv = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (sigma == NULL || sigma_inv == NULL)
    {
        info = SLIP_OUT_OF_MEMORY;
        goto done;
    }
    cycle_sigma (sigma, sigma_inv, n, k, m);

    // the update: in-place chain of adjacent diagonal pushes on the factors
    bool blk = false;
    int64_t blocked_at = -1;
    tic = clock ();
    info = distant_push_inplace (&L_s, &U_s, &rhos_s, L1, U1, rhos1, k, m,
        false, NULL, NULL, &blk, &blocked_at, option);
    toc = clock ();
    double t_upd = (double) (toc - tic) / CLOCKS_PER_SEC;
    bool blocked = (info == SLIP_SINGULAR);
    if (blocked) info = SLIP_OK;    // handled: the row reports the fallback
    BCK (info);

    // the refactorization: full SLIP pipeline (fresh COLAMD) on PAQ'
    BCK (build_paq_csc (&Ap, A, pinv1, q1, sigma, sigma_inv, option));
    tic = clock ();
    BCK (SLIP_LU_analyze (&Sr, Ap, option));
    BCK (SLIP_LU_factorize (&Lr, &Ur, &rr, &pr, Ap, Sr, option));
    toc = clock ();
    double t_ref = (double) (toc - tic) / CLOCKS_PER_SEC;

    // verification (never counted in the timings)
    const char *verify = "";
    if (!blocked)
    {
        bool det_ok = (mpz_cmpabs (rhos_s->x.mpz[n-1], rr->x.mpz[n-1]) == 0);

        // random-vector probe: PAQ' * v == L' * D'^(-1) * U' * v (exact).
        // Skipped when the update ended on a 2x2 block pivot: the probe
        // helper assumes a scalar diagonal D.
        bool probe_ok = true;
        if (!blk)
        {
            BCK (probe_LDU_eq_M_csc (&probe_ok, L_s, U_s, rhos_s, Ap,
                UINT64_C (0xC0FFEE1234567), option2));
        }

        BCK (SLIP_LU_analyze (&Sv, Ap, option2));
        BCK (SLIP_LU_factorize (&Lv, &Uv, &rv, &pv, Ap, Sv, option2));
        bool pv_identity = true;
        for (int64_t i = 0; i < n; i++)
        {
            if (pv[i] != i) { pv_identity = false; break; }
        }
        if (pv_identity)
        {
            bool exact = csc_equal (L_s, Lv) && csc_equal (U_s, Uv) &&
                equal_dense (rhos_s, rv);
            verify = (!det_ok || !exact || !probe_ok) ? "MISMATCH" :
                (blk ? "exact [2x2 block]" : "exact + probe");
        }
        else
        {
            verify = (!det_ok || !probe_ok) ? "MISMATCH" :
                (blk ? "det [2x2 block]" : "det + probe (row-pivoted)");
        }
    }

    // one table row
    int64_t d = (k < m) ? m - k : k - m;
    if (blocked)
    {
        printf ("  %5"PRId64" ->%6"PRId64" %6"PRId64"  %10s  %10.6f %9s"
            "  %9s %9"PRId64"   chain hits 0 pivot at %"PRId64
            " (refactor)\n", k + 1, m + 1, d, "--", t_ref, "--", "--",
            Lr->p[n] + Ur->p[n], blocked_at + 1);
    }
    else
    {
        char spd[32];
        if (t_upd > 0)
        {
            snprintf (spd, sizeof (spd), "%8.1fx", t_ref / t_upd);
        }
        else
        {
            snprintf (spd, sizeof (spd), "%9s", "inf");
        }
        printf ("  %5"PRId64" ->%6"PRId64" %6"PRId64"  %10.6f  %10.6f %s"
            "  %9"PRId64" %9"PRId64"   %s\n", k + 1, m + 1, d, t_upd, t_ref,
            spd, L_s->p[n] + U_s->p[n], Lr->p[n] + Ur->p[n], verify);
    }

done:
    SLIP_matrix_free (&L_s, option);
    SLIP_matrix_free (&U_s, option);
    SLIP_matrix_free (&rhos_s, option);
    SLIP_matrix_free (&Ap, option);
    SLIP_matrix_free (&Lr, option);
    SLIP_matrix_free (&Ur, option);
    SLIP_matrix_free (&rr, option);
    SLIP_matrix_free (&Lv, option);
    SLIP_matrix_free (&Uv, option);
    SLIP_matrix_free (&rv, option);
    SLIP_FREE (pr);
    SLIP_FREE (pv);
    SLIP_LU_analysis_free (&Sr, option);
    SLIP_LU_analysis_free (&Sv, option);
    SLIP_FREE (sigma);
    SLIP_FREE (sigma_inv);
    #undef BCK
    return info;
}

//------------------------------------------------------------------------------
// benchmark: column replacement (push to the end + one REF forward solve)
//------------------------------------------------------------------------------

/* Fill v[0..n-1] with a deterministic pseudorandom DENSE column: every
 * entry is in {-9..-1, 1..9} (never zero), so the incoming column has full
 * pattern.  Deterministic so runs are reproducible; the seed lets sequential
 * replacements (-RN) produce a distinct column at every step. */
static void fill_dense_column_seeded (mpz_t *v, int64_t n, uint64_t seed)
{
    uint64_t s = seed ? seed : UINT64_C (0x9E3779B97F4A7C15);
    for (int64_t r = 0; r < n; r++)
    {
        s = s * UINT64_C (6364136223846793005)
            + UINT64_C (1442695040888963407);
        int64_t val = (int64_t) ((s >> 33) % 9) + 1;    // 1..9
        if ((s >> 32) & 1) val = -val;
        mpz_set_si (v[r], val);
    }
}

static void fill_dense_column (mpz_t *v, int64_t n)
{
    fill_dense_column_seeded (v, n, UINT64_C (0x9E3779B97F4A7C15));
}

/* B' = A with column jcol replaced by the dense column v (CSC copy). */
static SLIP_info replace_col_csc (SLIP_matrix **A_out, const SLIP_matrix *A,
    int64_t jcol, const SLIP_matrix *v, const SLIP_options *option)
{
    SLIP_info info;
    int64_t n = A->n;
    int64_t nnz = A->p[n] - (A->p[jcol+1] - A->p[jcol]) + n;
    SLIP_matrix *C = NULL;
    info = SLIP_matrix_allocate (&C, SLIP_CSC, SLIP_MPZ, n, n, nnz,
        false, true, option);
    if (info != SLIP_OK) return info;
    int64_t nz = 0;
    for (int64_t j = 0; j < n; j++)
    {
        C->p[j] = nz;
        if (j == jcol)
        {
            for (int64_t r = 0; r < n; r++)
            {
                C->i[nz] = r;
                info = SLIP_mpz_set (C->x.mpz[nz], v->x.mpz[r]);
                if (info != SLIP_OK)
                {
                    SLIP_matrix_free (&C, option);
                    return info;
                }
                nz++;
            }
        }
        else
        {
            for (int64_t p = A->p[j]; p < A->p[j+1]; p++)
            {
                C->i[nz] = A->i[p];
                info = SLIP_mpz_set (C->x.mpz[nz], A->x.mpz[p]);
                if (info != SLIP_OK)
                {
                    SLIP_matrix_free (&C, option);
                    return info;
                }
                nz++;
            }
        }
    }
    C->p[n] = nz;
    *A_out = C;
    return SLIP_OK;
}

/* REF (IPGE) triangular forward solve for the LAST column of U:  x holds
 * the incoming column, dense, in the factorization's row space; on output
 * x[i] = U(i,n-1) and x[n-1] is the new final pivot rho[n-1].  Replacing
 * the last column changes nothing else: L columns 0..n-2 and U columns
 * 0..n-2 depend only on matrix columns 0..n-2, and L's last column is just
 * its diagonal (the new pivot).
 *
 * Follows slip_ref_triangular_solve's IPGE-with-history scheme: h[i] is the
 * last pivot step applied to x[i], and a history update
 * x[i] = x[i]*rho[j-1]/rho[h[i]] brings a stale entry to step j before use,
 * so the total work is O(nnz(L)) exact operations even though x is dense.
 * Every intermediate x[i] is a bordered minor of the replaced matrix
 * (integrality throughout).  Returns SLIP_SINGULAR iff x[n-1] = 0, i.e.
 * the replaced matrix is singular. */
static SLIP_info ref_solve_last (mpz_t *x, int64_t *h, const SLIP_matrix *L,
    mpz_t *rho, int64_t n)
{
    for (int64_t i = 0; i < n; i++) h[i] = -1;
    for (int64_t j = 0; j < n - 1; j++)
    {
        // finalize x[j]: this is U(j,n-1)
        if (mpz_sgn (x[j]) == 0) continue;
        if (h[j] < j - 1)
        {
            mpz_mul (x[j], x[j], rho[j-1]);
            if (h[j] > -1) mpz_divexact (x[j], x[j], rho[h[j]]);
        }
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            int64_t i = L->i[p];
            if (i <= j) continue;                   // diagonal
            if (mpz_sgn (L->x.mpz[p]) == 0) continue;
            if (mpz_sgn (x[i]) == 0)
            {
                // x[i] = ( 0 - L(i,j)*x[j] ) / rho[j-1]
                mpz_submul (x[i], L->x.mpz[p], x[j]);
                if (j > 0) mpz_divexact (x[i], x[i], rho[j-1]);
            }
            else
            {
                if (h[i] < j - 1)                   // history update
                {
                    mpz_mul (x[i], x[i], rho[j-1]);
                    if (h[i] > -1) mpz_divexact (x[i], x[i], rho[h[i]]);
                }
                // x[i] = ( rho[j]*x[i] - L(i,j)*x[j] ) / rho[j-1]
                mpz_mul (x[i], x[i], rho[j]);
                mpz_submul (x[i], L->x.mpz[p], x[j]);
                if (j > 0) mpz_divexact (x[i], x[i], rho[j-1]);
            }
            h[i] = j;
        }
    }
    // finalize x[n-1]: the new final pivot
    if (mpz_sgn (x[n-1]) != 0 && h[n-1] < n - 2)
    {
        mpz_mul (x[n-1], x[n-1], rho[n-2]);
        if (h[n-1] > -1) mpz_divexact (x[n-1], x[n-1], rho[h[n-1]]);
    }
    return (mpz_sgn (x[n-1]) == 0) ? SLIP_SINGULAR : SLIP_OK;
}

/* Replace basis column jcol (0-based) with a dense column and update the
 * factorization, timed against a full refactorization of the new basis B'.
 *
 * The update is the exact analogue of the Forrest-Tomlin basis exchange,
 * built from the primitives above:
 *
 *   1. MIXED DISTANT PUSH: the PAQ position cpos holding the leaving
 *      column (q1[cpos] = jcol) is pushed to the last position, in place,
 *      by the mixed column/diagonal chain (distant_push_inplace with
 *      mixed = true): each step takes the diagonal push, falling back to
 *      the column push exactly where the diagonal pivot vanishes (where
 *      the pure chain used to die) -- the upper support is then provably
 *      nonzero, so on a nonsingular basis the chain CANNOT hit a zero
 *      pivot and the refactorization fallback for blocked chains is gone.
 *      The column cycle is absorbed into Q; the row cycle is only the
 *      part contributed by the diagonal-push steps (rowmap), absorbed
 *      into P.  The factors then describe M = R*(PAQ)*sigma with the
 *      leaving column last.
 *   2. REF FORWARD SOLVE: replacing the LAST column needs no elimination.
 *      The new last column of U is the IPGE solve of the incoming column
 *      (mapped into M's row space through rowmap) through L, its last
 *      entry is the new final pivot, and L changes only in its trailing
 *      diagonal entry.
 *
 * The refactorization side runs the full SLIP pipeline (fresh COLAMD +
 * factorization) on B'.  Verified, outside the timings, by (a) the
 * determinant -- +-det(B') from the updated factors must equal the
 * refactorization's -- and (b) entrywise comparison with a fixed-order
 * refactorization of M with its last column replaced (factor uniqueness),
 * as in bench_push.
 *
 * The update path is reported unavailable (the refactorization stands as
 * the fallback) only if B' is singular, detected exactly by the forward
 * solve's zero final pivot.  (A zero-pivot abort of the chain is kept as a
 * defensive path but is unreachable for a nonsingular B.)
 */
static SLIP_info bench_replace (SLIP_matrix *A, SLIP_matrix *L1,
    SLIP_matrix *U1, SLIP_matrix *rhos1, const int64_t *pinv1,
    const int64_t *q1, int64_t jcol, double t_base, SLIP_options *option,
    SLIP_options *option2)
{
    SLIP_info info = SLIP_OK;
    int64_t n = A->n;
    SLIP_matrix *v = NULL, *xd = NULL, *Ap = NULL, *Mv = NULL;
    SLIP_matrix *L_s = NULL, *U_s = NULL, *rhos_s = NULL, *Un = NULL;
    SLIP_LU_analysis *Sr = NULL, *Sv = NULL;
    SLIP_matrix *Lr = NULL, *Ur = NULL, *rr = NULL;     // refactorization
    SLIP_matrix *Lv = NULL, *Uv = NULL, *rv = NULL;     // fixed-order verify
    int64_t *pr = NULL, *pv = NULL, *h = NULL;
    int64_t *sigma = NULL, *sigma_inv = NULL, *rowmap = NULL;
    clock_t tic, toc;

    #define RCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    // the leaving column's position in PAQ: q1[cpos] == jcol
    int64_t cpos = -1;
    for (int64_t c = 0; c < n; c++)
    {
        if (q1[c] == jcol) { cpos = c; break; }
    }
    if (cpos < 0) { info = SLIP_INCORRECT_INPUT; goto done; }

    printf ("\nReplacing basis column %"PRId64" (position %"PRId64" of PAQ) "
        "with a dense column\n(deterministic pseudorandom entries in "
        "{-9..-1, 1..9}).\n", jcol + 1, cpos + 1);

    // the incoming dense column and the replaced basis B'
    RCK (SLIP_matrix_allocate (&v, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, true, option));
    fill_dense_column (v->x.mpz, n);
    RCK (replace_col_csc (&Ap, A, jcol, v, option));

    sigma     = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    sigma_inv = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    rowmap    = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    h         = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (sigma == NULL || sigma_inv == NULL || rowmap == NULL || h == NULL)
    {
        info = SLIP_OUT_OF_MEMORY;
        goto done;
    }
    cycle_sigma (sigma, sigma_inv, n, cpos, n - 1);   // identity if cpos==n-1
    for (int64_t i = 0; i < n; i++) rowmap[i] = i;    // until the chain runs

    //--------------------------------------------------------------------------
    // the update: push the leaving position to the end, then one REF solve
    //--------------------------------------------------------------------------

    const char *skip = NULL;        // why the update path is unavailable
    bool upd_singular = false;      // the update detected B' is singular
    int64_t blocked_at = -1;
    int64_t ncolpush = 0;
    double t_push = 0, t_solve = 0;

    tic = clock ();
    if (cpos < n - 1)
    {
        info = distant_push_inplace (&L_s, &U_s, &rhos_s, L1, U1, rhos1,
            cpos, n - 1, true, rowmap, &ncolpush, NULL, &blocked_at, option);
        if (info == SLIP_SINGULAR)
        {
            // unreachable for a nonsingular B (the mixed chain always has
            // a feasible push); kept as a defensive fallback
            skip = "the push chain hits a zero pivot";
            info = SLIP_OK;
        }
        else
        {
            RCK (info);
        }
    }
    else        // the leaving column is already last: nothing to push
    {
        RCK (SLIP_matrix_copy (&L_s, SLIP_CSC, SLIP_MPZ, L1, option));
        RCK (SLIP_matrix_copy (&U_s, SLIP_CSC, SLIP_MPZ, U1, option));
        RCK (SLIP_matrix_copy (&rhos_s, SLIP_DENSE, SLIP_MPZ, rhos1,
            option));
    }
    toc = clock ();
    t_push = (double) (toc - tic) / CLOCKS_PER_SEC;

    if (skip == NULL)
    {
        tic = clock ();
        // the incoming column in M's row space: row r of B' lands at
        // position rowmap[pinv1[r]] (the chain's own row cycle -- only the
        // diagonal-push steps move rows)
        RCK (SLIP_matrix_allocate (&xd, SLIP_DENSE, SLIP_MPZ, n, 1, n,
            false, true, option));
        for (int64_t r = 0; r < n; r++)
        {
            mpz_set (xd->x.mpz[rowmap[pinv1[r]]], v->x.mpz[r]);
        }
        info = ref_solve_last (xd->x.mpz, h, L_s, rhos_s->x.mpz, n);
        if (info == SLIP_SINGULAR)
        {
            skip = "the replaced basis is singular";
            upd_singular = true;
            info = SLIP_OK;
        }
        else
        {
            RCK (info);
            // commit: new final pivot, L's trailing diagonal, U's last col
            mpz_set (rhos_s->x.mpz[n-1], xd->x.mpz[n-1]);
            for (int64_t p = L_s->p[n-1]; p < L_s->p[n]; p++)
            {
                if (L_s->i[p] == n - 1)
                {
                    mpz_set (L_s->x.mpz[p], xd->x.mpz[n-1]);
                }
            }
            int64_t base = U_s->p[n-1], xnz = 0;
            for (int64_t i = 0; i < n; i++)
            {
                if (mpz_sgn (xd->x.mpz[i]) != 0) xnz++;
            }
            RCK (SLIP_matrix_allocate (&Un, SLIP_CSC, SLIP_MPZ, n, n,
                base + xnz, false, true, option));
            for (int64_t j = 0; j < n - 1; j++) Un->p[j] = U_s->p[j];
            for (int64_t p = 0; p < base; p++)
            {
                Un->i[p] = U_s->i[p];
                mpz_swap (Un->x.mpz[p], U_s->x.mpz[p]);
            }
            Un->p[n-1] = base;
            int64_t nz = base;
            for (int64_t i = 0; i < n; i++)
            {
                if (mpz_sgn (xd->x.mpz[i]) != 0)
                {
                    Un->i[nz] = i;
                    mpz_swap (Un->x.mpz[nz], xd->x.mpz[i]);
                    nz++;
                }
            }
            Un->p[n] = nz;
            SLIP_matrix_free (&U_s, option);
            U_s = Un;
            Un = NULL;
        }
        toc = clock ();
        t_solve = (double) (toc - tic) / CLOCKS_PER_SEC;
    }

    //--------------------------------------------------------------------------
    // refactorization
    //--------------------------------------------------------------------------

    option->order = SLIP_NO_ORDERING;
    tic = clock ();
    RCK (SLIP_LU_analyze (&Sr, Ap, option));
    info = SLIP_LU_factorize (&Lr, &Ur, &rr, &pr, Ap, Sr, option);
    toc = clock ();
    double t_ref = (double) (toc - tic) / CLOCKS_PER_SEC;
    if (info == SLIP_SINGULAR)
    {
        printf ("\nThe refactorization reports the replaced basis is "
            "singular");
        if (upd_singular)
        {
            printf (" -- matching the update's finding.\n");
        }
        else if (skip != NULL)
        {
            printf (" (the update path was\nalready unavailable: %s, so it "
                "could not confirm this).\n", skip);
        }
        else
        {
            printf (".  ERROR: the update produced a NONZERO\nfinal pivot "
                "for a singular matrix -- this is a bug.\n");
        }
        info = SLIP_OK;
        goto done;
    }
    RCK (info);

    //--------------------------------------------------------------------------
    // verification (never counted in the timings)
    //--------------------------------------------------------------------------

    const char *verify = NULL;
    bool probe_ran = false, probe_ok = false;
    if (skip == NULL)
    {
        bool det_ok = (mpz_cmpabs (rhos_s->x.mpz[n-1], rr->x.mpz[n-1]) == 0);
        RCK (build_paq_csc (&Mv, Ap, pinv1, q1, sigma, rowmap, option2));

        // random-vector probe: PAQ' * v == L' * D'^(-1) * U' * v (exact).
        // Cheap and independent of the fixed-order refactor comparison below.
        RCK (probe_LDU_eq_M_csc (&probe_ok, L_s, U_s, rhos_s, Mv,
            UINT64_C (0xC0FFEE1234567), option2));
        probe_ran = true;

        RCK (SLIP_LU_analyze (&Sv, Mv, option2));
        RCK (SLIP_LU_factorize (&Lv, &Uv, &rv, &pv, Mv, Sv, option2));
        bool pv_identity = true;
        for (int64_t i = 0; i < n; i++)
        {
            if (pv[i] != i) { pv_identity = false; break; }
        }
        if (pv_identity)
        {
            bool exact = csc_equal (L_s, Lv) && csc_equal (U_s, Uv) &&
                equal_dense (rhos_s, rv);
            verify = (!det_ok || !exact) ? "MISMATCH" : "exact (factors "
                "match a fixed-order refactorization\n                "
                "entrywise, and the determinants agree)";
        }
        else
        {
            verify = !det_ok ? "MISMATCH" : "det only (the fixed-order "
                "refactorization row-pivoted)";
        }
    }

    //--------------------------------------------------------------------------
    // report
    //--------------------------------------------------------------------------

    printf ("\n---------------------------------------------------------------"
        "-\n");
    printf ("Column replacement: update vs. refactorization (CPU seconds)\n");
    printf ("-----------------------------------------------------------------"
        "\n");
    printf ("  %-48s %12.6f\n", "baseline factorization of B (context)",
        t_base);
    if (skip == NULL)
    {
        char lbl[128];
        snprintf (lbl, sizeof (lbl), "distant push %"PRId64" -> %"PRId64
            " (%"PRId64" column + %"PRId64" diagonal)", cpos + 1, n,
            ncolpush, (n - 1 - cpos) - ncolpush);
        printf ("  %-48s %12.6f\n", lbl, t_push);
        printf ("  %-48s %12.6f\n",
            "REF forward solve (new last column of U)", t_solve);
        printf ("  %-48s %12.6f\n", "total update", t_push + t_solve);
    }
    else
    {
        printf ("  update unavailable: %s;\n"
            "  the refactorization below is the fallback\n", skip);
        if (blocked_at >= 0)
        {
            printf ("  (chain stopped at position %"PRId64")\n",
                blocked_at + 1);
        }
    }
    printf ("  %-48s %12.6f\n",
        "refactorization of B' (COLAMD + factorize)", t_ref);
    printf ("-----------------------------------------------------------------"
        "\n");
    if (skip == NULL)
    {
        double t_upd = t_push + t_solve;
        if (t_upd > 0)
        {
            printf ("  update vs. refactorization: %.2fx\n", t_ref / t_upd);
        }
    }

    // sparsity: the nonzero count of every matrix involved, and the change
    // relative to the factors of B (did the update cost any sparsity?)
    int64_t fb = L1->p[n] + U1->p[n];
    int64_t fr = Lr->p[n] + Ur->p[n];
    if (skip == NULL)   // otherwise the timing table's rule is still open
    {
        printf ("-------------------------------------------------------------"
            "----\n");
    }
    printf ("Sparsity (nonzero counts; deltas are vs. the factors of B)\n");
    printf ("-----------------------------------------------------------------"
        "\n");
    printf ("  %-23s %38"PRId64"\n", "basis B", A->p[n]);
    printf ("  %-23s %38"PRId64"\n", "replaced basis B'", Ap->p[n]);
    printf ("  %-23s L %9"PRId64"   U %9"PRId64"   L+U %9"PRId64"\n",
        "factors of B", L1->p[n], U1->p[n], fb);
    if (skip == NULL)
    {
        int64_t fu = L_s->p[n] + U_s->p[n];
        printf ("  %-23s L %9"PRId64"   U %9"PRId64"   L+U %9"PRId64
            "  (%+"PRId64")\n", "updated factors of B'",
            L_s->p[n], U_s->p[n], fu, fu - fb);
        printf ("  %-23s L %9"PRId64"   U %9"PRId64"   L+U %9"PRId64
            "  (%+"PRId64")\n", "refactorized B'",
            Lr->p[n], Ur->p[n], fr, fr - fb);

        // machine-parseable delta (updated - refactorized); negative = update
        // is sparser than the fresh COLAMD refactor
        printf ("  update minus refactor nonzeros: %+"PRId64"\n", fu - fr);
        printf ("-------------------------------------------------------------"
            "----\n");
        printf ("  verification: %s\n", verify);
        if (probe_ran)
        {
            printf ("  random-vector probe: %s\n", probe_ok ? "MATCH" : "MISMATCH");
        }
    }
    else
    {
        printf ("  %-23s L %9"PRId64"   U %9"PRId64"   L+U %9"PRId64
            "  (%+"PRId64")\n", "refactorized B'",
            Lr->p[n], Ur->p[n], fr, fr - fb);
    }

done:
    SLIP_matrix_free (&v, option);
    SLIP_matrix_free (&xd, option);
    SLIP_matrix_free (&Ap, option);
    SLIP_matrix_free (&Mv, option);
    SLIP_matrix_free (&L_s, option);
    SLIP_matrix_free (&U_s, option);
    SLIP_matrix_free (&rhos_s, option);
    SLIP_matrix_free (&Un, option);
    SLIP_matrix_free (&Lr, option);
    SLIP_matrix_free (&Ur, option);
    SLIP_matrix_free (&rr, option);
    SLIP_matrix_free (&Lv, option);
    SLIP_matrix_free (&Uv, option);
    SLIP_matrix_free (&rv, option);
    SLIP_FREE (pr);
    SLIP_FREE (pv);
    SLIP_FREE (h);
    SLIP_FREE (sigma);
    SLIP_FREE (sigma_inv);
    SLIP_FREE (rowmap);
    SLIP_LU_analysis_free (&Sr, option);
    SLIP_LU_analysis_free (&Sv, option);
    #undef RCK
    return info;
}

//------------------------------------------------------------------------------
// sequential column replacement (-RN): fill trend over many updates
//------------------------------------------------------------------------------

/* Do nreps sequential column replacements on the basis A, each time updating
 * the tracked factors and (independently) refactoring from scratch for
 * comparison.  Prints one table row per replacement (time, fill, probe
 * verdict) and a summary at the end.
 *
 * we seed the dense pseudorandom column with k. distant push chain with mixed
 * pushes, one REF forward solve.
 *
 */
static SLIP_info bench_replace_seq (SLIP_matrix *A, SLIP_matrix *L1,
    SLIP_matrix *U1, SLIP_matrix *rhos1, const int64_t *pinv1,
    const int64_t *q1, int64_t jcol_start, int64_t nreps, double t_base,
    SLIP_options *option, SLIP_options *option2)
{
    SLIP_info info = SLIP_OK;
    int64_t n = A->n;

    // rolling state
    SLIP_matrix *A_cur = NULL;              // current basis, in original space
    SLIP_matrix *A_next = NULL;             // basis after this iteration
    SLIP_matrix *L = NULL, *U = NULL, *rhos = NULL;             // current factors
    SLIP_matrix *L_next = NULL, *U_next = NULL, *rhos_next = NULL;
    SLIP_matrix *Un = NULL;                 // scratch when appending last U col
    int64_t *pinv = NULL, *q = NULL;
    int64_t *pinv_next = NULL, *q_next = NULL;
    int64_t *qinv = NULL, *sigma = NULL, *sigma_inv = NULL;
    int64_t *rowmap = NULL, *h = NULL, *ident = NULL;

    // per-iteration scratch
    SLIP_matrix *v = NULL, *xd = NULL, *Mv = NULL;
    SLIP_matrix *Lr = NULL, *Ur = NULL, *rr = NULL;
    SLIP_LU_analysis *Sr = NULL;
    int64_t *pr = NULL;

    #define SQK(method) { info = (method); if (info != SLIP_OK) goto done; }

    // clone the initial factors and permutations so we can update them
    SQK (SLIP_matrix_copy (&A_cur, SLIP_CSC,   SLIP_MPZ, A,     option));
    SQK (SLIP_matrix_copy (&L,     SLIP_CSC,   SLIP_MPZ, L1,    option));
    SQK (SLIP_matrix_copy (&U,     SLIP_CSC,   SLIP_MPZ, U1,    option));
    SQK (SLIP_matrix_copy (&rhos,  SLIP_DENSE, SLIP_MPZ, rhos1, option));

    pinv      = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    q         = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    pinv_next = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    q_next    = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    qinv      = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    sigma     = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    sigma_inv = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    rowmap    = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    h         = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    ident     = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (!pinv || !q || !pinv_next || !q_next || !qinv || !sigma || !sigma_inv
        || !rowmap || !h || !ident)
    {
        info = SLIP_OUT_OF_MEMORY;
        goto done;
    }
    memcpy (pinv, pinv1, n * sizeof (int64_t));
    memcpy (q,    q1,    n * sizeof (int64_t));
    for (int64_t i = 0; i < n; i++) ident[i] = i;

    int64_t fb = L->p[n] + U->p[n];
    int64_t init_fb = fb;
    double t_upd_total = 0.0, t_ref_total = 0.0;
    int64_t bad = 0, done_reps = 0;
    int64_t final_fr = fb;

    printf ("\n---------------------------------------------------------------"
        "---------------------\n");
    printf ("Sequential column replacement: %"PRId64" iterations starting at "
        "column %"PRId64"\n", nreps, jcol_start + 1);
    printf ("(baseline factorization of B took %.6f s; L+U = %"PRId64")\n",
        t_base, fb);
    printf ("-----------------------------------------------------------------"
        "-------------------\n");
    printf ("  %4s %6s %6s %8s %8s %9s %9s %8s %s\n",
        "iter", "jcol", "cpos", "t_upd", "t_ref",
        "L+U(upd)", "L+U(ref)", "speedup", "verify");
    printf ("-----------------------------------------------------------------"
        "-------------------\n");

    for (int64_t k = 0; k < nreps; k++)
    {
        int64_t jcol = (jcol_start + k) % n;

        // cpos = position of jcol in the current column ordering
        for (int64_t c = 0; c < n; c++) qinv[q[c]] = c;
        int64_t cpos = qinv[jcol];

        // build the new dense column (per-iteration seed for distinct columns)
        SQK (SLIP_matrix_allocate (&v, SLIP_DENSE, SLIP_MPZ, n, 1, n,
            false, true, option));
        fill_dense_column_seeded (v->x.mpz, n,
            UINT64_C (0x9E3779B97F4A7C15) ^ (uint64_t) (k + 1));
        SQK (replace_col_csc (&A_next, A_cur, jcol, v, option));

        // permutation trackers for this step
        cycle_sigma (sigma, sigma_inv, n, cpos, n - 1);
        for (int64_t i = 0; i < n; i++) rowmap[i] = i;

        // the update: mixed distant push cpos -> n-1, then REF solve
        clock_t tic = clock ();
        bool singular = false;
        if (cpos < n - 1)
        {
            info = distant_push_inplace (&L_next, &U_next, &rhos_next, L, U,
                rhos, cpos, n - 1, true, rowmap, NULL, NULL, NULL, option);
            if (info == SLIP_SINGULAR) { singular = true; info = SLIP_OK; }
            else SQK (info);
        }
        else
        {
            SQK (SLIP_matrix_copy (&L_next,    SLIP_CSC,   SLIP_MPZ, L,    option));
            SQK (SLIP_matrix_copy (&U_next,    SLIP_CSC,   SLIP_MPZ, U,    option));
            SQK (SLIP_matrix_copy (&rhos_next, SLIP_DENSE, SLIP_MPZ, rhos, option));
        }

        if (!singular)
        {
            SQK (SLIP_matrix_allocate (&xd, SLIP_DENSE, SLIP_MPZ, n, 1, n,
                false, true, option));
            for (int64_t r = 0; r < n; r++)
            {
                mpz_set (xd->x.mpz[rowmap[pinv[r]]], v->x.mpz[r]);
            }
            info = ref_solve_last (xd->x.mpz, h, L_next, rhos_next->x.mpz, n);
            if (info == SLIP_SINGULAR) { singular = true; info = SLIP_OK; }
            else SQK (info);
        }

        if (!singular)
        {
            // commit new last-column entries into L, U, rhos
            mpz_set (rhos_next->x.mpz[n-1], xd->x.mpz[n-1]);
            for (int64_t p = L_next->p[n-1]; p < L_next->p[n]; p++)
            {
                if (L_next->i[p] == n - 1)
                {
                    mpz_set (L_next->x.mpz[p], xd->x.mpz[n-1]);
                }
            }
            int64_t base = U_next->p[n-1], xnz = 0;
            for (int64_t i = 0; i < n; i++)
            {
                if (mpz_sgn (xd->x.mpz[i]) != 0) xnz++;
            }
            SQK (SLIP_matrix_allocate (&Un, SLIP_CSC, SLIP_MPZ, n, n,
                base + xnz, false, true, option));
            for (int64_t j = 0; j < n - 1; j++) Un->p[j] = U_next->p[j];
            for (int64_t p = 0; p < base; p++)
            {
                Un->i[p] = U_next->i[p];
                mpz_swap (Un->x.mpz[p], U_next->x.mpz[p]);
            }
            Un->p[n-1] = base;
            int64_t nz = base;
            for (int64_t i = 0; i < n; i++)
            {
                if (mpz_sgn (xd->x.mpz[i]) != 0)
                {
                    Un->i[nz] = i;
                    mpz_swap (Un->x.mpz[nz], xd->x.mpz[i]);
                    nz++;
                }
            }
            Un->p[n] = nz;
            SLIP_matrix_free (&U_next, option);
            U_next = Un;
            Un = NULL;
        }
        clock_t toc = clock ();
        double t_upd = (double) (toc - tic) / CLOCKS_PER_SEC;

        // refactor for comparison (never counted against the update)
        tic = clock ();
        option->order = SLIP_NO_ORDERING;
        SQK (SLIP_LU_analyze (&Sr, A_next, option));
        info = SLIP_LU_factorize (&Lr, &Ur, &rr, &pr, A_next, Sr, option);
        toc = clock ();
        double t_ref = (double) (toc - tic) / CLOCKS_PER_SEC;
        bool ref_singular = (info == SLIP_SINGULAR);
        if (ref_singular) info = SLIP_OK;
        else SQK (info);

        // compose new permutations for the next iteration
        for (int64_t c = 0; c < n; c++) q_next[c]    = q[sigma[c]];
        for (int64_t r = 0; r < n; r++) pinv_next[r] = rowmap[pinv[r]];

        const char *verdict;
        int64_t fu = singular ? -1 : L_next->p[n] + U_next->p[n];
        int64_t fr = ref_singular ? -1 : Lr->p[n] + Ur->p[n];

        if (singular && ref_singular)
        {
            verdict = "singular (both agree)";
        }
        else if (singular != ref_singular)
        {
            verdict = "SINGULAR MISMATCH";
            bad++;
        }
        else
        {
            // probe against the target M defined by the composed permutations
            SQK (build_paq_csc (&Mv, A_next, pinv_next, q_next, ident, NULL,
                option2));
            bool probe_ok = false;
            SQK (probe_LDU_eq_M_csc (&probe_ok, L_next, U_next, rhos_next,
                Mv, UINT64_C (0xC0FFEE1234567) + (uint64_t) k, option2));
            SLIP_matrix_free (&Mv, option2);
            bool det_ok = (mpz_cmpabs (rhos_next->x.mpz[n-1],
                rr->x.mpz[n-1]) == 0);
            if (probe_ok && det_ok) verdict = "ok";
            else { verdict = "MISMATCH"; bad++; }
        }

        char spd[16];
        if (!singular && !ref_singular && t_upd > 0)
        {
            snprintf (spd, sizeof (spd), "%6.2fx", t_ref / t_upd);
        }
        else
        {
            snprintf (spd, sizeof (spd), "%7s", "--");
        }
        char fu_s[32], fr_s[32];
        if (fu < 0) snprintf (fu_s, sizeof (fu_s), "%9s", "SING");
        else snprintf (fu_s, sizeof (fu_s), "%9"PRId64, fu);
        if (fr < 0) snprintf (fr_s, sizeof (fr_s), "%9s", "SING");
        else snprintf (fr_s, sizeof (fr_s), "%9"PRId64, fr);

        printf ("  %4"PRId64" %6"PRId64" %6"PRId64" %8.4f %8.4f %s %s %s %s\n",
            k + 1, jcol + 1, cpos + 1, t_upd, t_ref, fu_s, fr_s, spd, verdict);

        if (!singular && !ref_singular)
        {
            t_upd_total += t_upd;
            t_ref_total += t_ref;
            done_reps++;
            final_fr = fr;
            fb = fu;
        }

        // rotate state: (A_cur, L, U, rhos, pinv, q) := (A_next, L_next, ...)
        SLIP_matrix_free (&A_cur, option); A_cur = A_next; A_next = NULL;
        SLIP_matrix_free (&L, option);     L    = L_next;    L_next = NULL;
        SLIP_matrix_free (&U, option);     U    = U_next;    U_next = NULL;
        SLIP_matrix_free (&rhos, option);  rhos = rhos_next; rhos_next = NULL;
        { int64_t *t = pinv; pinv = pinv_next; pinv_next = t; }
        { int64_t *t = q;    q    = q_next;    q_next    = t; }

        SLIP_matrix_free (&Lr, option);
        SLIP_matrix_free (&Ur, option);
        SLIP_matrix_free (&rr, option);
        SLIP_FREE (pr); pr = NULL;
        SLIP_LU_analysis_free (&Sr, option);
        SLIP_matrix_free (&v, option);
        SLIP_matrix_free (&xd, option);

        if (singular || ref_singular) break;
    }

    printf ("-----------------------------------------------------------------"
        "-------------------\n");
    printf ("Summary over %"PRId64" successful iterations "
        "(started at L+U = %"PRId64"):\n", done_reps, init_fb);
    if (done_reps > 0)
    {
        printf ("  total update time:       %10.6f  (avg %.6f per step)\n",
            t_upd_total, t_upd_total / (double) done_reps);
        printf ("  total refactor time:     %10.6f  (avg %.6f per step)\n",
            t_ref_total, t_ref_total / (double) done_reps);
        printf ("  cumulative speedup:      %10.2fx\n",
            (t_upd_total > 0) ? (t_ref_total / t_upd_total) : 0.0);
        printf ("  final L+U (update):      %10"PRId64"  (%+"PRId64
            " vs. baseline)\n", fb, fb - init_fb);
        printf ("  final L+U (refactor):    %10"PRId64"  (%+"PRId64
            " vs. baseline)\n", final_fr, final_fr - init_fb);
        printf ("  update vs. refactor:     %+"PRId64" (%s by %.2f%%)\n",
            fb - final_fr, (fb < final_fr) ? "sparser" : "denser",
            100.0 * (double) (fb < final_fr ? final_fr - fb : fb - final_fr)
                / (double) final_fr);
    }
    printf ("  verification failures:   %10"PRId64"\n", bad);

done:
    SLIP_matrix_free (&A_cur,     option);
    SLIP_matrix_free (&A_next,    option);
    SLIP_matrix_free (&L,         option);
    SLIP_matrix_free (&U,         option);
    SLIP_matrix_free (&rhos,      option);
    SLIP_matrix_free (&L_next,    option);
    SLIP_matrix_free (&U_next,    option);
    SLIP_matrix_free (&rhos_next, option);
    SLIP_matrix_free (&Un,        option);
    SLIP_matrix_free (&v,         option);
    SLIP_matrix_free (&xd,        option);
    SLIP_matrix_free (&Mv,        option);
    SLIP_matrix_free (&Lr,        option);
    SLIP_matrix_free (&Ur,        option);
    SLIP_matrix_free (&rr,        option);
    SLIP_FREE (pinv);
    SLIP_FREE (q);
    SLIP_FREE (pinv_next);
    SLIP_FREE (q_next);
    SLIP_FREE (qinv);
    SLIP_FREE (sigma);
    SLIP_FREE (sigma_inv);
    SLIP_FREE (rowmap);
    SLIP_FREE (h);
    SLIP_FREE (ident);
    SLIP_FREE (pr);
    SLIP_LU_analysis_free (&Sr, option);
    #undef SQK
    return info;
}

//------------------------------------------------------------------------------
// reconstruction helper
//------------------------------------------------------------------------------

/* Reconstruct the (un-permuted) matrix from its factors:  compute
 * M = L * D^(-1) * U exactly in rational (MPQ) arithmetic, then apply the
 * inverse permutations so that
 *
 *      A_rec[i][j] = M[pinv[i]][qinv[j]]
 *
 * which should equal the matrix that was factored.  On success *Arec_handle
 * points to a freshly allocated n-by-n dense MPZ matrix.
 *
 * Permutation conventions (from the SLIP_LU sources):
 *   - column ordering Q:  column c of A*Q is column q[c] of A.
 *   - row permutation P:  P has a 1 at (pinv[i], i), so
 *                         (P*A*Q)[pinv[i]][c] = A[i][q[c]].
 * Hence A[i][q[c]] = M[pinv[i]][c]; with j = q[c] (c = qinv[j]) this gives the
 * formula above.
 *
 * kb >= 0 marks a 2x2 block pivot at positions {kb, kb+1} (see apdpu): D
 * then carries a 2x2 block there, and rows kb, kb+1 of D^(-1)*U are formed
 * with the true 2x2 inverse (adjugate over determinant) instead of the
 * scalar reciprocals.  Pass kb = -1 for an all-scalar D.
 */
static SLIP_info reconstruct (SLIP_matrix **Arec_handle,
    const SLIP_matrix *L_dense, const SLIP_matrix *U_dense,
    const SLIP_matrix *D, const int64_t *pinv, const int64_t *q,
    int64_t n, int64_t kb, const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    SLIP_matrix *M = NULL;      // M = L * D^(-1) * U         (rational)
    SLIP_matrix *W = NULL;      // W = D^(-1) * U             (rational, scratch)
    SLIP_matrix *A_rec = NULL;  // reconstructed matrix       (integer)
    SLIP_matrix *tmpq = NULL;   // scratch MPQ scalars
    int64_t *qinv = NULL;       // inverse of the column permutation q

    #define RCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    // build qinv, the inverse of the column permutation q
    qinv = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (qinv == NULL) { info = SLIP_OUT_OF_MEMORY; goto done; }
    for (int64_t c = 0; c < n; c++) qinv[q[c]] = c;

    RCK (SLIP_matrix_allocate (&M, SLIP_DENSE, SLIP_MPQ, n, n, n*n,
        false, true, option));
    RCK (SLIP_matrix_allocate (&W, SLIP_DENSE, SLIP_MPQ, n, n, n*n,
        false, true, option));
    RCK (SLIP_matrix_allocate (&A_rec, SLIP_DENSE, SLIP_MPZ, n, n, n*n,
        false, true, option));
    RCK (SLIP_matrix_allocate (&tmpq, SLIP_DENSE, SLIP_MPQ, 8, 1, 8,
        false, true, option));

    // named scratch scalars within tmpq
    mpq_t *one  = &SLIP_1D (tmpq, 0, mpq);  // the constant 1
    mpq_t *dkk  = &SLIP_1D (tmpq, 1, mpq);  // D(k,k) as a rational
    mpq_t *dinv = &SLIP_1D (tmpq, 2, mpq);  // 1 / D(k,k)
    mpq_t *lval = &SLIP_1D (tmpq, 3, mpq);  // an entry of L as a rational
    mpq_t *prod = &SLIP_1D (tmpq, 4, mpq);  // running product
    mpq_t *acc  = &SLIP_1D (tmpq, 5, mpq);  // running sum
    mpq_t *det  = &SLIP_1D (tmpq, 6, mpq);  // det of the 2x2 D block
    mpq_t *uval = &SLIP_1D (tmpq, 7, mpq);  // an entry of U as a rational

    RCK (SLIP_mpq_set_ui (*one, 1, 1));

    // W = D^(-1) * U : scale row k of U by 1/D(k,k)
    for (int64_t k = 0; k < n; k++)
    {
        if (kb >= 0 && (k == kb || k == kb + 1)) continue;  // block rows below
        RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, k, k, mpz)));
        RCK (SLIP_mpq_div (*dinv, *one, *dkk));
        for (int64_t c = 0; c < n; c++)
        {
            RCK (SLIP_mpq_set_z (*prod, SLIP_2D (U_dense, k, c, mpz)));
            RCK (SLIP_mpq_mul (SLIP_2D (W, k, c, mpq), *prod, *dinv));
        }
    }

    if (kb >= 0)
    {
        // rows kb, kb+1 of W = (2x2 D block)^(-1) * (rows kb, kb+1 of U),
        // via the adjugate:  inv = [ d11 -d01 ; -d10 d00 ] / det
        RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb, kb, mpz)));
        RCK (SLIP_mpq_set_z (*dinv, SLIP_2D (D, kb+1, kb+1, mpz)));
        RCK (SLIP_mpq_mul (*det, *dkk, *dinv));             // d00*d11
        RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb, kb+1, mpz)));
        RCK (SLIP_mpq_set_z (*dinv, SLIP_2D (D, kb+1, kb, mpz)));
        RCK (SLIP_mpq_mul (*prod, *dkk, *dinv));            // d01*d10
        mpq_sub (*det, *det, *prod);    // SLIP_mpq_sub is not wrapped
        for (int64_t c = 0; c < n; c++)
        {
            RCK (SLIP_mpq_set_z (*uval, SLIP_2D (U_dense, kb, c, mpz)));
            RCK (SLIP_mpq_set_z (*lval, SLIP_2D (U_dense, kb+1, c, mpz)));

            // W(kb,c) = ( d11*U(kb,c) - d01*U(kb+1,c) ) / det
            RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb+1, kb+1, mpz)));
            RCK (SLIP_mpq_mul (*acc, *dkk, *uval));
            RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb, kb+1, mpz)));
            RCK (SLIP_mpq_mul (*prod, *dkk, *lval));
            mpq_sub (*acc, *acc, *prod);
            RCK (SLIP_mpq_div (SLIP_2D (W, kb, c, mpq), *acc, *det));

            // W(kb+1,c) = ( d00*U(kb+1,c) - d10*U(kb,c) ) / det
            RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb, kb, mpz)));
            RCK (SLIP_mpq_mul (*acc, *dkk, *lval));
            RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, kb+1, kb, mpz)));
            RCK (SLIP_mpq_mul (*prod, *dkk, *uval));
            mpq_sub (*acc, *acc, *prod);
            RCK (SLIP_mpq_div (SLIP_2D (W, kb+1, c, mpq), *acc, *det));
        }
    }

    // M = L * W
    for (int64_t r = 0; r < n; r++)
    {
        for (int64_t c = 0; c < n; c++)
        {
            RCK (SLIP_mpq_set_ui (*acc, 0, 1));
            for (int64_t k = 0; k < n; k++)
            {
                RCK (SLIP_mpq_set_z (*lval, SLIP_2D (L_dense, r, k, mpz)));
                RCK (SLIP_mpq_mul (*prod, *lval, SLIP_2D (W, k, c, mpq)));
                RCK (SLIP_mpq_add (*acc, *acc, *prod));
            }
            RCK (SLIP_mpq_set (SLIP_2D (M, r, c, mpq), *acc));
        }
    }

    // un-permute M into A_rec: A_rec[i][j] = M[pinv[i]][qinv[j]]
    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            RCK (SLIP_mpz_set_q (SLIP_2D (A_rec, i, j, mpz),
                SLIP_2D (M, pinv[i], qinv[j], mpq)));
        }
    }

    *Arec_handle = A_rec;   // hand ownership to the caller
    A_rec = NULL;

done:
    SLIP_matrix_free (&M, option);
    SLIP_matrix_free (&W, option);
    SLIP_matrix_free (&A_rec, option);
    SLIP_matrix_free (&tmpq, option);
    SLIP_FREE (qinv);
    #undef RCK
    return info;
}

//------------------------------------------------------------------------------
// probe helper: random-vector exactness check of L * D^(-1) * U == M
//------------------------------------------------------------------------------

/* Exact "am I really this matrix" test that avoids the O(n^3) mpq blowup of
 * reconstruct().  Draws a deterministic pseudorandom integer vector v (entries
 * in [-2^30, 2^30)) and checks
 *
 *      M * v  ==  L * D^(-1) * U * v
 *
 * with M, L, U in CSC MPZ and rhos in DENSE MPZ; D(k,k) = rhos[k-1]*rhos[k],
 * rhos[-1] = 1 (no 2x2-block support here -- the block cases are diagonal
 * pushes, not what -R uses).  Cost is O(nnz(M) + nnz(U) + nnz(L)) GMP ops
 * plus an elementwise D^(-1) scaling; no dense n x n intermediate.
 *
 * The check is probabilistic in principle: a wrong pair with rational error
 * matrix E = L*D^(-1)*U - M would need E*v = 0 exactly on the drawn v, an
 * event whose probability for the 31-bit v drawn here is at most (max |E|
 * entry as a rational)/2^30 -- vanishingly small in practice.  Call twice with
 * different seeds to squeeze that probability further if desired.
 */
static SLIP_info probe_LDU_eq_M_csc (bool *ok, const SLIP_matrix *L,
    const SLIP_matrix *U, const SLIP_matrix *rhos, const SLIP_matrix *M,
    uint64_t seed, const SLIP_options *option)
{
    SLIP_info info = SLIP_OK;
    int64_t n = L->n;
    SLIP_matrix *v = NULL, *w1 = NULL, *y = NULL;   // integer vectors
    SLIP_matrix *z = NULL, *w2 = NULL;              // rational vectors
    SLIP_matrix *tmpq = NULL;                       // scratch mpq scalars
    mpz_t dkk;
    bool dkk_init = false;

    #define PCK(method) { info = (method); if (info != SLIP_OK) goto done; }

    *ok = false;

    PCK (SLIP_matrix_allocate (&v,  SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, true, option));
    PCK (SLIP_matrix_allocate (&w1, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, true, option));
    PCK (SLIP_matrix_allocate (&y,  SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, true, option));
    PCK (SLIP_matrix_allocate (&z,  SLIP_DENSE, SLIP_MPQ, n, 1, n,
        false, true, option));
    PCK (SLIP_matrix_allocate (&w2, SLIP_DENSE, SLIP_MPQ, n, 1, n,
        false, true, option));
    PCK (SLIP_matrix_allocate (&tmpq, SLIP_DENSE, SLIP_MPQ, 2, 1, 2,
        false, true, option));
    mpz_init (dkk);
    dkk_init = true;

    mpq_t *t1 = &SLIP_1D (tmpq, 0, mpq);
    mpq_t *t2 = &SLIP_1D (tmpq, 1, mpq);

    // draw v with a splitmix64-style PRNG; entries in [-2^30, 2^30)
    uint64_t s = seed ? seed : UINT64_C (0x9E3779B97F4A7C15);
    for (int64_t i = 0; i < n; i++)
    {
        s += UINT64_C (0x9E3779B97F4A7C15);
        uint64_t r = s;
        r = (r ^ (r >> 30)) * UINT64_C (0xBF58476D1CE4E5B9);
        r = (r ^ (r >> 27)) * UINT64_C (0x94D049BB133111EB);
        r ^= (r >> 31);
        int64_t vi = (int64_t) (r & UINT64_C (0x7FFFFFFF)) - INT64_C (0x40000000);
        mpz_set_si (SLIP_1D (v, i, mpz), vi);
    }

    // w1 = M * v
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
        {
            mpz_addmul (SLIP_1D (w1, M->i[p], mpz),
                M->x.mpz[p], SLIP_1D (v, j, mpz));
        }
    }

    // y = U * v
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            mpz_addmul (SLIP_1D (y, U->i[p], mpz),
                U->x.mpz[p], SLIP_1D (v, j, mpz));
        }
    }

    // z[k] = y[k] / D(k,k) with D(k,k) = rhos[k-1] * rhos[k], rhos[-1] = 1
    for (int64_t k = 0; k < n; k++)
    {
        if (k == 0)
        {
            mpz_set (dkk, rhos->x.mpz[0]);
        }
        else
        {
            PCK (SLIP_mpz_mul (dkk, rhos->x.mpz[k-1], rhos->x.mpz[k]));
        }
        if (mpz_sgn (dkk) == 0) { info = SLIP_SINGULAR; goto done; }
        mpq_set_num (SLIP_1D (z, k, mpq), SLIP_1D (y, k, mpz));
        mpq_set_den (SLIP_1D (z, k, mpq), dkk);
        mpq_canonicalize (SLIP_1D (z, k, mpq));
    }

    // w2 = L * z  (rational SpMV, walking CSC L by columns)
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            PCK (SLIP_mpq_set_z (*t1, L->x.mpz[p]));
            PCK (SLIP_mpq_mul (*t2, *t1, SLIP_1D (z, j, mpq)));
            PCK (SLIP_mpq_add (SLIP_1D (w2, L->i[p], mpq),
                SLIP_1D (w2, L->i[p], mpq), *t2));
        }
    }

    // compare: w1[i] (integer) == w2[i] (rational)
    bool eq = true;
    for (int64_t i = 0; i < n && eq; i++)
    {
        PCK (SLIP_mpq_set_z (*t1, SLIP_1D (w1, i, mpz)));
        if (!mpq_equal (*t1, SLIP_1D (w2, i, mpq))) eq = false;
    }
    *ok = eq;

done:
    SLIP_matrix_free (&v,    option);
    SLIP_matrix_free (&w1,   option);
    SLIP_matrix_free (&y,    option);
    SLIP_matrix_free (&z,    option);
    SLIP_matrix_free (&w2,   option);
    SLIP_matrix_free (&tmpq, option);
    if (dkk_init) mpz_clear (dkk);
    #undef PCK
    return info;
}

//------------------------------------------------------------------------------
// main
//------------------------------------------------------------------------------

#define FREE_WORKSPACE                           \
    SLIP_matrix_free(&A, option);                \
    SLIP_matrix_free(&A_dense, option);          \
    SLIP_matrix_free(&L1, option);               \
    SLIP_matrix_free(&U1, option);               \
    SLIP_matrix_free(&L1_dense, option);         \
    SLIP_matrix_free(&U1_dense, option);         \
    SLIP_matrix_free(&D1, option);               \
    SLIP_matrix_free(&A_rec1, option);           \
    SLIP_matrix_free(&rhos1, option);            \
    SLIP_FREE(pinv1);                            \
    SLIP_LU_analysis_free(&S1, option);          \
    SLIP_matrix_free(&PAQ, option);              \
    SLIP_matrix_free(&PAQ_csc, option);          \
    SLIP_matrix_free(&L_upd, option);            \
    SLIP_matrix_free(&U_upd, option);            \
    SLIP_matrix_free(&rhos_upd, option);         \
    SLIP_matrix_free(&D_upd, option);            \
    SLIP_matrix_free(&A_rec_upd, option);        \
    SLIP_FREE(iperm);                            \
    SLIP_FREE(sigma);                            \
    SLIP_FREE(sigma_inv);                        \
    SLIP_matrix_free(&L_s, option);              \
    SLIP_matrix_free(&U_s, option);              \
    SLIP_matrix_free(&rhos_s, option);           \
    SLIP_matrix_free(&Ls_dense, option);         \
    SLIP_matrix_free(&Us_dense, option);         \
    SLIP_matrix_free(&L2, option);               \
    SLIP_matrix_free(&U2, option);               \
    SLIP_matrix_free(&L2_dense, option);         \
    SLIP_matrix_free(&U2_dense, option);         \
    SLIP_matrix_free(&D2, option);               \
    SLIP_matrix_free(&A_rec2, option);           \
    SLIP_matrix_free(&rhos2, option);            \
    SLIP_FREE(pinv2);                            \
    SLIP_LU_analysis_free(&S2, option);          \
    SLIP_FREE(option);                           \
    SLIP_FREE(option2);                          \
    SLIP_finalize( ) ;

int main (int argc, char* argv[])
{

    //--------------------------------------------------------------------------
    // Initialize the SLIP LU environment.
    //--------------------------------------------------------------------------

    SLIP_initialize();

    //--------------------------------------------------------------------------
    // Declare and initialize the data structures used below.
    //--------------------------------------------------------------------------

    SLIP_matrix *A = NULL;          // user input matrix (CSC, MPZ)
    SLIP_matrix *A_dense = NULL;    // dense copy of A (for forming PAQ)

    // factorization 1 (of A, with COLAMD)
    SLIP_LU_analysis *S1 = NULL;
    SLIP_matrix *L1 = NULL, *U1 = NULL, *rhos1 = NULL;
    SLIP_matrix *L1_dense = NULL, *U1_dense = NULL, *D1 = NULL, *A_rec1 = NULL;
    int64_t *pinv1 = NULL;

    // PAQ (the matrix SLIP actually factors) and its swapped/CSC forms
    SLIP_matrix *PAQ = NULL;        // dense PAQ, then swapped in place
    SLIP_matrix *PAQ_csc = NULL;    // CSC copy of the swapped PAQ

    // dense-APCPU-updated factors of the swapped PAQ
    SLIP_matrix *L_upd = NULL, *U_upd = NULL, *rhos_upd = NULL, *D_upd = NULL;
    SLIP_matrix *A_rec_upd = NULL;
    int64_t *iperm = NULL;          // identity permutation, for reconstruct
    int64_t *sigma = NULL, *sigma_inv = NULL;   // position permutation of PAQ'

    // sparse-APCPU-updated factors (CSC) and their dense copies for the
    // dense-vs-sparse cross-check
    SLIP_matrix *L_s = NULL, *U_s = NULL, *rhos_s = NULL;
    SLIP_matrix *Ls_dense = NULL, *Us_dense = NULL;

    // factorization 2 (of the swapped PAQ, no ordering + diagonal pivoting),
    // used as the from-scratch reference the update is compared against
    SLIP_LU_analysis *S2 = NULL;
    SLIP_matrix *L2 = NULL, *U2 = NULL, *rhos2 = NULL;
    SLIP_matrix *L2_dense = NULL, *U2_dense = NULL, *D2 = NULL, *A_rec2 = NULL;
    int64_t *pinv2 = NULL;

    // Options.  'option' uses the defaults (COLAMD, tolerance pivoting) for the
    // first factorization; 'option2' factors the swapped PAQ as-is.
    SLIP_options *option  = SLIP_create_default_options();
    SLIP_options *option2 = SLIP_create_default_options();
    if (!option || !option2)
    {
        fprintf (stderr, "Error! OUT of MEMORY!\n");
        SLIP_FREE (option);
        SLIP_FREE (option2);
        SLIP_finalize();
        return 0;
    }
    option2->order = SLIP_NO_ORDERING;  // keep the manual column swap (Q = I)
    option2->pivot = SLIP_DIAGONAL;     // avoid extra row pivoting (P ~ I)
    SLIP_info ok;

    // timing (CPU seconds, measured with clock() as in SLIPLU.c); the table
    // is printed after everything else has completed
    double t_analyze1 = 0, t_factor1 = 0, t_apcpu = 0, t_sparse = 0;
    double t_analyze2 = 0, t_factor2 = 0;
    clock_t tic, toc;

    // Command line: perm [-Q] [-D] [matrix_file [k]], or
    //               perm -B [-Q] matrix_file [k m], or
    //               perm -R [-Q] [matrix_file [col]].
    // -Q selects the QSOpt_ex basis format (default: SLIP triplet format);
    // -D performs the adjacent DIAGONAL push (symmetric swap of columns AND
    // rows k, k+1) instead of the column push; -B runs the distant diagonal
    // push benchmark (chained APDPU vs. full refactorization); -R replaces
    // one basis column with a dense column and benchmarks the update
    // (distant push to the end + one REF forward solve) against a full
    // refactorization of the new basis.
    bool qsx = false, diag = false, bench = false, repl = false;
    int64_t nreps = 0;      // -R alone: 0 (single, detailed).  -R<N>: N reps.
    int ai = 1;
    while (ai < argc && argv[ai][0] == '-')
    {
        if      (strcmp (argv[ai], "-Q") == 0) { qsx = true; ai++; }
        else if (strcmp (argv[ai], "-D") == 0) { diag = true; ai++; }
        else if (strcmp (argv[ai], "-B") == 0) { bench = true; ai++; }
        else if (argv[ai][0] == '-' && argv[ai][1] == 'R')
        {
            // -R (one replacement, detailed report) or -R<N> (N sequential
            // replacements, table + summary)
            repl = true;
            const char *tail = argv[ai] + 2;
            if (*tail != '\0')
            {
                char *end = NULL;
                long v = strtol (tail, &end, 10);
                if (end == tail || *end != '\0' || v < 1)
                {
                    fprintf (stderr, "invalid -R count '%s' (need positive "
                        "integer)\n", tail);
                    FREE_WORKSPACE;
                    return 0;
                }
                nreps = (int64_t) v;
            }
            ai++;
        }
        else
        {
            fprintf (stderr, "unknown flag '%s'\n"
                "usage: perm [-Q] [-D] [matrix_file [k]]\n"
                "       perm -B [-Q] matrix_file [k m]\n"
                "       perm -R[N] [-Q] [matrix_file [col]]\n", argv[ai]);
            FREE_WORKSPACE;
            return 0;
        }
    }
    if (bench && repl)
    {
        fprintf (stderr, "-B and -R are mutually exclusive\n");
        FREE_WORKSPACE;
        return 0;
    }

    // -R defaults to the cycle basis snapshot (the format sniffer below
    // recognizes it as a QSOpt_ex basis without needing -Q)
    char *mat_name = repl ? "cycle_Bases/basis_k0_B.txt"
                          : "../ExampleMats/test_mat2.txt";
    if (ai < argc)
    {
        mat_name = argv[ai++];
    }
    else if (qsx && !repl)
    {
        fprintf (stderr, "usage: perm -Q <basis_file> [k]\n");
        FREE_WORKSPACE;
        return 0;
    }
    char *k_arg = (ai < argc) ? argv[ai++] : NULL;
    char *m_arg = (ai < argc) ? argv[ai] : NULL;

    //--------------------------------------------------------------------------
    // Read in the matrix A (SLIP triplet format, or a QSOpt_ex basis if -Q).
    //--------------------------------------------------------------------------

    FILE* mat_file = fopen(mat_name, "r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }

    // Sniff the format before trusting the -Q flag.  All three formats
    // share the "m n nnz" header; they differ from the second line on:
    //
    //   old QSOpt_ex basis:  a single per-row denominator (1 token);
    //   new QSOpt_ex basis:  0-indexed triplets whose values may be
    //                        rationals "p/q" (3 tokens);
    //   SLIP triplet file:   1-indexed triplets with numeric values
    //                        (3 tokens, never a '/' or a 0 index).
    //
    // A 3-token second line is therefore a new-format basis iff a row or
    // column index 0 appears (SLIP triplets are 1-based) or a '/' shows up
    // in the first stretch of entries.  Misreading a basis as triplets is
    // fatal: the 0-indexed triplets get the 1-based decrement and the
    // resulting -1 indices crash SLIP_matrix_copy.
    bool qsx_rational = false;
    {
        char line[1024];
        double h1, h2, h3;
        long r, c;
        char val[256];
        if (fgets (line, sizeof (line), mat_file) != NULL &&
            sscanf (line, "%lf %lf %lf", &h1, &h2, &h3) == 3 &&
            fgets (line, sizeof (line), mat_file) != NULL)
        {
            int ntok = sscanf (line, "%ld %ld %255s", &r, &c, val);
            if (ntok == 1)
            {
                if (!qsx)
                {
                    fprintf (stderr, "%s looks like an old-format QSOpt_ex "
                        "basis (single denominator after the header); "
                        "reading it with the -Q reader\n", mat_name);
                    qsx = true;
                }
            }
            else if (ntok == 3)
            {
                bool basis = (r == 0 || c == 0 ||
                    strchr (val, '/') != NULL);
                for (int64_t s = 0; !basis && s < 100 &&
                    fgets (line, sizeof (line), mat_file) != NULL; s++)
                {
                    basis = (strchr (line, '/') != NULL);
                }
                if (basis)
                {
                    qsx_rational = true;
                    if (!qsx)
                    {
                        fprintf (stderr, "%s looks like a rational QSOpt_ex "
                            "basis (0-indexed / rational triplets); reading "
                            "it with the -Q reader\n", mat_name);
                        qsx = true;
                    }
                }
                else if (qsx)
                {
                    fprintf (stderr, "%s looks like a SLIP triplet file "
                        "(1-indexed numeric triplets); ignoring -Q\n",
                        mat_name);
                    qsx = false;
                }
            }
        }
        rewind (mat_file);
    }

    if (qsx)
    {
        OK(read_qsx_basis(&A, mat_file, qsx_rational, option));
    }
    else
    {
        OK(SLIP_tripread(&A, mat_file, option));
    }
    fclose(mat_file);

    int64_t n = A->n;
    printf ("Read %s%s: %"PRId64"-by-%"PRId64", nnz = %"PRId64"\n",
        mat_name, qsx ? " (QSOpt_ex basis)" : "", A->m, A->n, A->p[A->n]);

    // The dense parts of the demo (dense APCPU, grids, reconstructions) need
    // about a dozen n-by-n mpz matrices; disable them when that would
    // exhaust memory.  The sparse APCPU and its verification against a
    // from-scratch refactorization run at any size.
    double dense_gb = 12.0 * (double) n * (double) n * 16.0 / 1e9;
    bool dense_ok = (dense_gb <= 2.0);
    if (!dense_ok)
    {
        printf ("Dense workspace for n = %"PRId64" would need roughly "
            "%.0f GB; running the\nsparse-only workflow (no dense update, "
            "grids, or reconstructions).\n", n, dense_gb);
    }
    if (bench || repl) dense_ok = false;    // the benchmarks are sparse-only

    if (dense_ok)
    {
        OK(SLIP_matrix_copy(&A_dense, SLIP_DENSE, SLIP_MPZ, A, option));
        print_dense_gated ("A matrix (original input)", A_dense);
    }

    //--------------------------------------------------------------------------
    // Factorization 1: factor A normally (COLAMD + tolerance pivoting) to get
    // the column ordering Q (= S1->q) and the row permutation P (= pinv1).
    //--------------------------------------------------------------------------

    tic = clock();
    OK(SLIP_LU_analyze(&S1, A, option));
    toc = clock();
    t_analyze1 = (double) (toc - tic) / CLOCKS_PER_SEC;

    tic = clock();
    OK(SLIP_LU_factorize(&L1, &U1, &rhos1, &pinv1, A, S1, option));
    toc = clock();
    t_factor1 = (double) (toc - tic) / CLOCKS_PER_SEC;

    //--------------------------------------------------------------------------
    // Benchmark mode (-B): distant diagonal pushes vs. full refactorization.
    //--------------------------------------------------------------------------

    if (bench)
    {
        printf ("\nBenchmark: distant diagonal push (chained sparse APDPU) "
            "vs. full refactorization.\nBaseline factorization of A: "
            "analysis %.6f s + factorization %.6f s,\nnnz(L) + nnz(U) = "
            "%"PRId64".  Each row moves row/column k of PAQ to position m\n"
            "(cyclic rotation of the positions in between, 1-based); the "
            "refactorization\ncolumn runs the full SLIP pipeline (fresh "
            "COLAMD + factorization) on the\npermuted matrix.\n",
            t_analyze1, t_factor1, L1->p[n] + U1->p[n]);
        printf ("\n  %5s ->%6s %6s  %10s  %10s %9s  %9s %9s   %s\n",
            "k", "m", "d", "update(s)", "refact(s)", "speedup",
            "nnz(upd)", "nnz(ref)", "verification");
        printf ("  -----------------------------------------------------"
            "----------------------------------\n");

        if (k_arg != NULL && m_arg != NULL)
        {
            int64_t kb = (int64_t) atol (k_arg) - 1;
            int64_t mb = (int64_t) atol (m_arg) - 1;
            if (kb < 0 || kb >= n || mb < 0 || mb >= n || kb == mb)
            {
                fprintf (stderr, "invalid push %s -> %s (need two distinct "
                    "positions in 1..%"PRId64")\n", k_arg, m_arg, n);
                FREE_WORKSPACE;
                return 0;
            }
            OK (bench_push (A, L1, U1, rhos1, pinv1, S1->q, kb, mb,
                option, option2));
        }
        else
        {
            // sweep: distances 1, 2, 4, ... (centered), then the two
            // full-length pushes 1 -> n and n -> 1
            int64_t dmax = 0;
            for (int64_t d = 1; d < n; d *= 2)
            {
                OK (bench_push (A, L1, U1, rhos1, pinv1, S1->q,
                    (n - 1 - d) / 2, (n - 1 - d) / 2 + d, option, option2));
                dmax = d;
            }
            if (dmax != n - 1)
            {
                OK (bench_push (A, L1, U1, rhos1, pinv1, S1->q, 0, n - 1,
                    option, option2));
            }
            OK (bench_push (A, L1, U1, rhos1, pinv1, S1->q, n - 1, 0,
                option, option2));
        }

        printf ("\nverification (never counted in the timings): 'exact' = "
            "the updated factors\nmatch a fixed-order refactorization of the "
            "permuted matrix entrywise (factor\nuniqueness) AND the "
            "determinants agree; 'det' = determinant check only (the\n"
            "fixed-order refactorization had to row-pivot, so its factors "
            "are not\ncomparable entrywise).\n");
        FREE_WORKSPACE;
        return 0;
    }

    //--------------------------------------------------------------------------
    // Replacement mode (-R): swap one basis column for a dense column and
    // benchmark the update (push to the end + REF solve) vs. refactorization.
    //--------------------------------------------------------------------------

    if (repl)
    {
        int64_t jc;
        if (k_arg != NULL)
        {
            jc = (int64_t) atol (k_arg) - 1;    // 1-based on the command line
            if (jc < 0 || jc >= n)
            {
                fprintf (stderr, "invalid column %s (need 1..%"PRId64")\n",
                    k_arg, n);
                FREE_WORKSPACE;
                return 0;
            }
        }
        else
        {
            jc = ask_col (n);
            if (jc < 0) { FREE_WORKSPACE; return 0; }
        }
        if (nreps > 0)
        {
            OK (bench_replace_seq (A, L1, U1, rhos1, pinv1, S1->q, jc, nreps,
                t_analyze1 + t_factor1, option, option2));
        }
        else
        {
            OK (bench_replace (A, L1, U1, rhos1, pinv1, S1->q, jc,
                t_analyze1 + t_factor1, option, option2));
        }
        FREE_WORKSPACE;
        return 0;
    }

    if (dense_ok)
    {
        OK(SLIP_matrix_copy(&L1_dense, SLIP_DENSE, SLIP_MPZ, L1, option));
        OK(SLIP_matrix_copy(&U1_dense, SLIP_DENSE, SLIP_MPZ, U1, option));
        OK(build_D(&D1, rhos1, n, -1, NULL, option));

        print_dense_gated ("L (factorization 1: A with COLAMD)", L1_dense);
        print_dense_gated ("U (factorization 1: A with COLAMD)", U1_dense);
        print_dense_gated ("D (factorization 1: A with COLAMD)", D1);

        // sanity check: reconstructing from factorization 1 returns A
        if (n <= RECON_MAX_N)
        {
            OK(reconstruct(&A_rec1, L1_dense, U1_dense, D1, pinv1, S1->q, n,
                -1, option));
            printf ("\nA reconstructed from factorization 1: %s\n",
                equal_dense (A_rec1, A_dense) ?
                "MATCHES A" : "DOES NOT MATCH A");
        }
        else
        {
            printf ("\nA reconstruction check skipped (n = %"PRId64" > %d)\n",
                n, RECON_MAX_N);
        }

        //----------------------------------------------------------------------
        // Form PAQ explicitly: the matrix SLIP actually factored above.
        //     (P*A*Q)[pinv1[i]][c] = A[i][q1[c]]
        //----------------------------------------------------------------------

        OK(SLIP_matrix_allocate(&PAQ, SLIP_DENSE, SLIP_MPZ, n, n, n*n,
            false, true, option));
        for (int64_t i = 0; i < n; i++)
        {
            for (int64_t c = 0; c < n; c++)
            {
                OK(SLIP_mpz_set(SLIP_2D(PAQ, pinv1[i], c, mpz),
                    SLIP_2D(A_dense, i, S1->q[c], mpz)));
            }
        }
        print_dense_gated ("PAQ (the matrix factorization 1 actually factored)",
            PAQ);
    }

    //--------------------------------------------------------------------------
    // Ask which ADJACENT columns of PAQ to swap (k on the command line wins),
    // then swap columns kc and kc+1 of PAQ in place (Step 0 of APCPU).
    //--------------------------------------------------------------------------

    int64_t kc;
    if (k_arg != NULL)
    {
        kc = (int64_t) atol (k_arg) - 1;        // 1-based on the command line
        if (kc < 0 || kc > n - 2)
        {
            fprintf (stderr, "invalid k = %s (need 1..%"PRId64")\n",
                k_arg, n - 1);
            FREE_WORKSPACE;
            return 0;
        }
    }
    else
    {
        kc = ask_k (n);
        if (kc < 0) { FREE_WORKSPACE; return 0; }
    }
    if (diag)
    {
        printf ("\nDiagonal push: swapping adjacent columns AND rows "
            "%"PRId64" and %"PRId64" of PAQ (1-based).\n", kc + 1, kc + 2);
    }
    else
    {
        printf ("\nSwapping adjacent columns %"PRId64" and %"PRId64" of PAQ "
            "(1-based).\n", kc + 1, kc + 2);
    }

    if (dense_ok)
    {
        swap_cols (PAQ, kc, kc + 1);
        if (diag) swap_rows (PAQ, kc, kc + 1);
        print_dense_gated (diag ?
            "PAQ' (PAQ after the symmetric swap)" :
            "PAQ' (PAQ after the adjacent column swap)", PAQ);
    }

    //--------------------------------------------------------------------------
    // Sparse update (APCPU or APDPU): update the CSC factors directly,
    // using only stored entries of L1, U1, and rhos1 (back-solve trick; PAQ
    // is not consulted).  This also screens the singular case up front.
    //--------------------------------------------------------------------------

    bool blk_sparse = false, blk_dense = false;    // 2x2 block pivot used?
    tic = clock();
    if (diag)
    {
        ok = apdpu_sparse (&L_s, &U_s, &rhos_s, L1, U1, rhos1, kc,
            &blk_sparse, option);
    }
    else
    {
        ok = apcpu_sparse (&L_s, &U_s, &rhos_s, L1, U1, rhos1, kc, option);
    }
    toc = clock();
    t_sparse = (double) (toc - tic) / CLOCKS_PER_SEC;
    if (ok == SLIP_SINGULAR)    // only the column push can still land here
    {
        printf ("\n%s: new pivot at position %"PRId64" is zero; the %s "
            "PAQ has no REF LU\nfactorization with this row permutation "
            "(row pivoting would be required).\n",
            diag ? "APDPU" : "APCPU", kc + 1,
            diag ? "symmetrically swapped" : "swapped");
        FREE_WORKSPACE;
        return 0;
    }
    OK (ok);
    if (blk_sparse)
    {
        printf ("\nAPDPU: new pivot at position %"PRId64" is zero (the "
            "leading %"PRId64"-by-%"PRId64" block of PAQ' is singular,\n"
            "though PAQ' itself is not); falling back to a 2x2 BLOCK PIVOT "
            "at positions\n%"PRId64",%"PRId64".  The block determinant "
            "rho[%"PRId64"]*rho[%"PRId64"] is nonzero, so this cannot fail;\n"
            "D' carries a 2x2 block there and rhos'[%"PRId64"] stays 0 as "
            "the block marker.\n",
            kc + 1, kc + 1, kc + 1, kc + 1, kc + 2, kc, kc + 2, kc + 1);
    }
    printf ("\nSparse %s done: nnz(L') = %"PRId64", nnz(U') = %"PRId64
        " (old: %"PRId64", %"PRId64")%s\n", diag ? "APDPU" : "APCPU",
        L_s->p[n], U_s->p[n], L1->p[n], U1->p[n],
        blk_sparse ? " [2x2 block pivot]" : "");

    //--------------------------------------------------------------------------
    // Dense update: APCPU (column push) or APDPU (diagonal push), then
    // cross-check against the sparse implementation where both ran.
    //--------------------------------------------------------------------------

    if (dense_ok)
    {
        tic = clock();
        if (diag)
        {
            ok = apdpu (&L_upd, &U_upd, &rhos_upd, L1_dense, U1_dense,
                rhos1, kc, &blk_dense, option);
        }
        else
        {
            ok = apcpu (&L_upd, &U_upd, &rhos_upd, L1_dense, U1_dense,
                rhos1, PAQ, kc, option);
        }
        toc = clock();
        t_apcpu = (double) (toc - tic) / CLOCKS_PER_SEC;
        OK (ok);        // sparse update already screened SLIP_SINGULAR
        if (blk_dense != blk_sparse)
        {
            printf ("\nERROR: sparse and dense APDPU disagree on the 2x2 "
                "block pivot fallback\n");
        }
        OK (build_D (&D_upd, rhos_upd, n, blk_dense ? kc : -1, L_upd,
            option));

        print_dense_gated (diag ? "L' (APDPU-updated)" : "L' (APCPU-updated)",
            L_upd);
        print_dense_gated (diag ? "U' (APDPU-updated)" : "U' (APCPU-updated)",
            U_upd);
        print_dense_gated (diag ? "D' (APDPU-updated)" : "D' (APCPU-updated)",
            D_upd);

        // exactness check: L' * D'^(-1) * U' must equal the swapped PAQ
        if (n <= RECON_MAX_N)
        {
            iperm = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
            if (iperm == NULL) { FREE_WORKSPACE; return 0; }
            for (int64_t i = 0; i < n; i++) iperm[i] = i;
            OK (reconstruct (&A_rec_upd, L_upd, U_upd, D_upd, iperm, iperm, n,
                blk_dense ? kc : -1, option));
            printf ("\nL' * D'^(-1) * U' reconstruction: %s\n",
                equal_dense (A_rec_upd, PAQ) ?
                "MATCHES the swapped PAQ" : "DOES NOT MATCH the swapped PAQ");
        }
        else
        {
            printf ("\nL' * D'^(-1) * U' reconstruction check skipped "
                "(n = %"PRId64" > %d)\n", n, RECON_MAX_N);
        }

        // random-vector probe: PAQ * v == L' * D'^(-1) * U' * v (exact).
        // Runs at any n and is not gated on RECON_MAX_N; skipped only when
        // the update ended on a 2x2 block pivot, since the probe helper
        // assumes a scalar diagonal D.
        if (!blk_dense)
        {
            SLIP_matrix *PAQ_probe = NULL;
            OK (SLIP_matrix_copy (&PAQ_probe, SLIP_CSC, SLIP_MPZ, PAQ, option));
            bool probe_ok = false;
            OK (probe_LDU_eq_M_csc (&probe_ok, L_s, U_s, rhos_s, PAQ_probe,
                UINT64_C (0xC0FFEE1234567), option));
            SLIP_matrix_free (&PAQ_probe, option);
            printf ("Random-vector probe:"
                " %s\n", probe_ok ? "MATCH" : "MISMATCH");
        }

        // dense and sparse updates must produce identical factors
        OK (SLIP_matrix_copy (&Ls_dense, SLIP_DENSE, SLIP_MPZ, L_s, option));
        OK (SLIP_matrix_copy (&Us_dense, SLIP_DENSE, SLIP_MPZ, U_s, option));
        printf ("\nsparse %s vs. dense %s:\n", diag ? "APDPU" : "APCPU",
            diag ? "APDPU" : "APCPU");
        printf ("  L:    %s\n", equal_dense (Ls_dense, L_upd) ?
            "MATCH" : "MISMATCH");
        printf ("  U:    %s\n", equal_dense (Us_dense, U_upd) ?
            "MATCH" : "MISMATCH");
        printf ("  rhos: %s\n", equal_dense (rhos_s, rhos_upd) ?
            "MATCH" : "MISMATCH");
    }

    //--------------------------------------------------------------------------
    // Reference: re-factor the swapped PAQ from scratch with NO column
    // ordering (identity Q, so the swap is preserved) and diagonal pivoting,
    // and compare the factors with the APCPU-updated ones entrywise.
    //--------------------------------------------------------------------------

    sigma     = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    sigma_inv = (int64_t *) SLIP_malloc (n * sizeof (int64_t));
    if (sigma == NULL || sigma_inv == NULL) { FREE_WORKSPACE; return 0; }
    cycle_sigma (sigma, sigma_inv, n, kc, kc + 1);  // adjacent transposition
    OK(build_paq_csc(&PAQ_csc, A, pinv1, S1->q, sigma,
        diag ? sigma_inv : NULL, option2));

    tic = clock();
    OK(SLIP_LU_analyze(&S2, PAQ_csc, option2));
    toc = clock();
    t_analyze2 = (double) (toc - tic) / CLOCKS_PER_SEC;

    tic = clock();
    OK(SLIP_LU_factorize(&L2, &U2, &rhos2, &pinv2, PAQ_csc, S2, option2));
    toc = clock();
    t_factor2 = (double) (toc - tic) / CLOCKS_PER_SEC;

    if (dense_ok)
    {
        OK(SLIP_matrix_copy(&L2_dense, SLIP_DENSE, SLIP_MPZ, L2, option2));
        OK(SLIP_matrix_copy(&U2_dense, SLIP_DENSE, SLIP_MPZ, U2, option2));
        OK(build_D(&D2, rhos2, n, -1, NULL, option2));

        print_dense_gated ("L (from-scratch refactorization of swapped PAQ)",
            L2_dense);
        print_dense_gated ("U (from-scratch refactorization of swapped PAQ)",
            U2_dense);
        print_dense_gated ("D (from-scratch refactorization of swapped PAQ)",
            D2);

        // reconstruct: should return the swapped PAQ handed to factorization 2
        if (n <= RECON_MAX_N)
        {
            OK(reconstruct(&A_rec2, L2_dense, U2_dense, D2, pinv2, S2->q, n,
                -1, option2));
            printf ("\nL * D^(-1) * U reconstruction (from scratch): %s\n",
                equal_dense (A_rec2, PAQ) ?
                "MATCHES the swapped PAQ" : "DOES NOT MATCH the swapped PAQ");
        }
        else
        {
            printf ("\nL * D^(-1) * U reconstruction check (from scratch) "
                "skipped (n = %"PRId64" > %d)\n", n, RECON_MAX_N);
        }
    }

    // The factors are unique for fixed P and Q, so if the from-scratch
    // refactorization kept P = identity (diagonal pivoting succeeded on every
    // step), the APCPU factors must match it entrywise.
    bool p2_identity = true;
    for (int64_t i = 0; i < n; i++)
    {
        if (pinv2[i] != i) { p2_identity = false; break; }
    }
    if (p2_identity)
    {
        if (dense_ok)
        {
            printf ("\ndense %s factors vs. from-scratch "
                "refactorization:\n", diag ? "APDPU" : "APCPU");
            printf ("  L:    %s\n", equal_dense (L_upd, L2_dense) ?
                "MATCH" : "MISMATCH");
            printf ("  U:    %s\n", equal_dense (U_upd, U2_dense) ?
                "MATCH" : "MISMATCH");
            printf ("  D:    %s\n", equal_dense (D_upd, D2) ?
                "MATCH" : "MISMATCH");
            printf ("  rhos: %s\n", equal_dense (rhos_upd, rhos2) ?
                "MATCH" : "MISMATCH");
        }
        printf ("\nsparse %s factors vs. from-scratch refactorization:\n",
            diag ? "APDPU" : "APCPU");
        printf ("  L:    %s\n", csc_equal (L_s, L2) ? "MATCH" : "MISMATCH");
        printf ("  U:    %s\n", csc_equal (U_s, U2) ? "MATCH" : "MISMATCH");
        printf ("  rhos: %s\n", equal_dense (rhos_s, rhos2) ?
            "MATCH" : "MISMATCH");
    }
    else
    {
        printf ("\nNote: the from-scratch refactorization used a nontrivial "
            "row permutation%s,\nso its factors are not directly comparable "
            "with the updated factors\n(the reconstruction checks above are "
            "the authoritative test%s).\n",
            blk_sparse ? " (forced by the zero diagonal pivot)" : "",
            blk_sparse ? ", plus the sparse-vs-dense cross-check" : "");
    }

    //--------------------------------------------------------------------------
    // Timing summary.
    //--------------------------------------------------------------------------

    printf ("\n----------------------------------------------------------------\n");
    printf ("Timing summary (CPU seconds)\n");
    printf ("----------------------------------------------------------------\n");
    printf ("  %-48s %12.6f\n", "symbolic analysis of A (COLAMD)", t_analyze1);
    printf ("  %-48s %12.6f\n", "factorization of A (SLIP LU)", t_factor1);
    printf ("  %-48s %12.6f\n", diag ?
        "sparse APDPU update of the factors" :
        "sparse APCPU update of the factors", t_sparse);
    if (dense_ok)
    {
        printf ("  %-48s %12.6f\n", diag ?
            "dense APDPU update of the factors" :
            "dense APCPU update of the factors", t_apcpu);
    }
    printf ("  %-48s %12.6f\n", "symbolic analysis of swapped PAQ", t_analyze2);
    printf ("  %-48s %12.6f\n", "from-scratch factorization of swapped PAQ",
        t_factor2);
    printf ("----------------------------------------------------------------\n");
    if (t_sparse > 0)
    {
        printf ("  sparse update vs. from-scratch refactorization: %.2fx\n",
            t_factor2 / t_sparse);
    }
    else
    {
        printf ("  sparse update vs. from-scratch refactorization: update\n"
                "  below clock() resolution\n");
    }
    if (dense_ok && t_apcpu > 0)
    {
        printf ("  dense  update vs. from-scratch refactorization: %.2fx\n",
            t_factor2 / t_apcpu);
    }

    //--------------------------------------------------------------------------
    // Free memory.
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    return 0;
}
