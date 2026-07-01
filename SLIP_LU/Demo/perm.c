//------------------------------------------------------------------------------
// SLIP_LU/Demo/perm.c: SLIP LU factorization, experimenting with permutations
//------------------------------------------------------------------------------

#include "SLIP_LU.h"
#include "demos.h"
#include <stdio.h>

/* This program reads a sparse matrix stored in triplet format and performs the
 * SLIP LU factorization PAQ = L*D^(-1)*U twice, to show how a permutation of
 * the *already permuted* matrix PAQ affects the factorization:
 *
 *   1. Factor A "normally" (COLAMD column ordering + tolerance pivoting) to
 *      obtain the column ordering Q and the row permutation P.
 *   2. Form the matrix that SLIP actually factors, PAQ, explicitly.
 *   3. Ask the user for two columns of PAQ and two rows of PAQ to swap.
 *   4. Re-factor the swapped PAQ with NO column ordering (identity Q) and
 *      diagonal pivoting, so the manual swap is exactly what gets factored,
 *      and print the resulting L, U, and D.
 *
 * Usage:
 *      perm [matrix_file]
 *
 * If no matrix file is given on the command line, the matrix is read from
 * "../ExampleMats/test_mat2.txt"
 */

//------------------------------------------------------------------------------
// printing / swapping helpers
//------------------------------------------------------------------------------

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

/* Prompt the user for a pair of 1-based indices in [1,n] to swap, returning
 * them 0-based in *a and *b.  On equal indices, invalid input, or an
 * out-of-range value, sets *a == *b so the caller performs no swap.  Prompts
 * are written to stderr so they do not pollute the matrices printed to stdout.
 */
static void ask_swap (const char *what, int64_t n, int64_t *a, int64_t *b)
{
    long x = 0, y = 0;
    fprintf (stderr, "Enter two %s to swap (1..%"PRId64", same = no swap): ",
        what, n);
    if (scanf ("%ld %ld", &x, &y) != 2 ||
        x < 1 || x > n || y < 1 || y > n)
    {
        fprintf (stderr, "  (no valid pair given; leaving %s unchanged)\n",
            what);
        *a = *b = 0;
        return;
    }
    *a = x - 1;     // convert to 0-based
    *b = y - 1;
}

//------------------------------------------------------------------------------
// D construction and reconstruction helpers (return SLIP_info)
//------------------------------------------------------------------------------

/* Build the diagonal matrix D from the pivot sequence rhos.  D is all zeros
 * except on the diagonal, where D(k,k) is the product of consecutive pivots:
 *
 *      D(k,k) = rhos[k-1] * rhos[k],   with the convention rhos[-1] = 1.
 *
 * So D(0,0) = rhos[0], D(1,1) = rhos[0]*rhos[1], D(2,2) = rhos[1]*rhos[2], ...
 * These are the pivots that make PAQ = L * D^(-1) * U.  On success *D_handle
 * points to a freshly allocated n-by-n dense MPZ matrix.
 */
static SLIP_info build_D (SLIP_matrix **D_handle, const SLIP_matrix *rhos,
    int64_t n, const SLIP_options *option)
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

    *D_handle = D;
    return SLIP_OK;
}

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
 */
static SLIP_info reconstruct (SLIP_matrix **Arec_handle,
    const SLIP_matrix *L_dense, const SLIP_matrix *U_dense,
    const SLIP_matrix *D, const int64_t *pinv, const int64_t *q,
    int64_t n, const SLIP_options *option)
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
    RCK (SLIP_matrix_allocate (&tmpq, SLIP_DENSE, SLIP_MPQ, 6, 1, 6,
        false, true, option));

    // named scratch scalars within tmpq
    mpq_t *one  = &SLIP_1D (tmpq, 0, mpq);  // the constant 1
    mpq_t *dkk  = &SLIP_1D (tmpq, 1, mpq);  // D(k,k) as a rational
    mpq_t *dinv = &SLIP_1D (tmpq, 2, mpq);  // 1 / D(k,k)
    mpq_t *lval = &SLIP_1D (tmpq, 3, mpq);  // an entry of L as a rational
    mpq_t *prod = &SLIP_1D (tmpq, 4, mpq);  // running product
    mpq_t *acc  = &SLIP_1D (tmpq, 5, mpq);  // running sum

    RCK (SLIP_mpq_set_ui (*one, 1, 1));

    // W = D^(-1) * U : scale row k of U by 1/D(k,k)
    for (int64_t k = 0; k < n; k++)
    {
        RCK (SLIP_mpq_set_z (*dkk, SLIP_2D (D, k, k, mpz)));
        RCK (SLIP_mpq_div (*dinv, *one, *dkk));
        for (int64_t c = 0; c < n; c++)
        {
            RCK (SLIP_mpq_set_z (*prod, SLIP_2D (U_dense, k, c, mpz)));
            RCK (SLIP_mpq_mul (SLIP_2D (W, k, c, mpq), *prod, *dinv));
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

    // factorization 2 (of the swapped PAQ, no ordering + diagonal pivoting)
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

    // Matrix file: command-line argument if given, else the default test matrix.
    char *mat_name = "../ExampleMats/test_mat2.txt";
    if (argc > 1) mat_name = argv[1];

    //--------------------------------------------------------------------------
    // Read in the matrix A from the triplet-format file.
    //--------------------------------------------------------------------------

    FILE* mat_file = fopen(mat_name, "r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SLIP_tripread(&A, mat_file, option));
    fclose(mat_file);

    int64_t n = A->n;
    OK(SLIP_matrix_copy(&A_dense, SLIP_DENSE, SLIP_MPZ, A, option));
    print_dense_grid ("A matrix (original input)", A_dense);

    //--------------------------------------------------------------------------
    // Factorization 1: factor A normally (COLAMD + tolerance pivoting) to get
    // the column ordering Q (= S1->q) and the row permutation P (= pinv1).
    //--------------------------------------------------------------------------

    OK(SLIP_LU_analyze(&S1, A, option));
    OK(SLIP_LU_factorize(&L1, &U1, &rhos1, &pinv1, A, S1, option));

    OK(SLIP_matrix_copy(&L1_dense, SLIP_DENSE, SLIP_MPZ, L1, option));
    OK(SLIP_matrix_copy(&U1_dense, SLIP_DENSE, SLIP_MPZ, U1, option));
    OK(build_D(&D1, rhos1, n, option));

    print_dense_grid ("L (factorization 1: A with COLAMD)", L1_dense);
    print_dense_grid ("U (factorization 1: A with COLAMD)", U1_dense);
    print_dense_grid ("D (factorization 1: A with COLAMD)", D1);

    // sanity check: reconstructing from factorization 1 returns A
    OK(reconstruct(&A_rec1, L1_dense, U1_dense, D1, pinv1, S1->q, n, option));
    print_dense_grid ("A reconstructed from factorization 1 (should equal A)",
        A_rec1);

    //--------------------------------------------------------------------------
    // Form PAQ explicitly: the matrix SLIP actually factored above.
    //     (P*A*Q)[pinv1[i]][c] = A[i][q1[c]]
    //--------------------------------------------------------------------------

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
    print_dense_grid ("PAQ (the matrix factorization 1 actually factored)", PAQ);

    //--------------------------------------------------------------------------
    // Ask the user which two columns and two rows of PAQ to swap, then apply
    // the swaps to PAQ in place.  Equal/invalid indices mean "no swap".
    //--------------------------------------------------------------------------

    int64_t c1, c2, r1, r2;
    ask_swap ("columns of PAQ", n, &c1, &c2);
    ask_swap ("rows of PAQ",    n, &r1, &r2);

    swap_cols (PAQ, c1, c2);
    swap_rows (PAQ, r1, r2);
    print_dense_grid ("PAQ after swapping the chosen columns and rows", PAQ);

    //--------------------------------------------------------------------------
    // Factorization 2: re-factor the swapped PAQ with NO column ordering
    // (identity Q, so the swap is preserved) and diagonal pivoting.
    //--------------------------------------------------------------------------

    OK(SLIP_matrix_copy(&PAQ_csc, SLIP_CSC, SLIP_MPZ, PAQ, option2));
    OK(SLIP_LU_analyze(&S2, PAQ_csc, option2));
    OK(SLIP_LU_factorize(&L2, &U2, &rhos2, &pinv2, PAQ_csc, S2, option2));

    OK(SLIP_matrix_copy(&L2_dense, SLIP_DENSE, SLIP_MPZ, L2, option2));
    OK(SLIP_matrix_copy(&U2_dense, SLIP_DENSE, SLIP_MPZ, U2, option2));
    OK(build_D(&D2, rhos2, n, option2));

    print_dense_grid ("L (factorization 2: swapped PAQ)", L2_dense);
    print_dense_grid ("U (factorization 2: swapped PAQ)", U2_dense);
    print_dense_grid ("D (factorization 2: swapped PAQ)", D2);

    // reconstruct: should return the swapped PAQ we handed to factorization 2
    OK(reconstruct(&A_rec2, L2_dense, U2_dense, D2, pinv2, S2->q, n, option2));
    print_dense_grid (
        "swapped PAQ reconstructed from factorization 2 (should equal it)",
        A_rec2);

    //--------------------------------------------------------------------------
    // Free memory.
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    return 0;
}
