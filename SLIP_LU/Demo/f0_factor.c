#include "SLIP_LU.h"
#include "demos.h"

// take a matrix Adense, compute the slip factorization of it
// also capable of taking -C and -D flags to perform an adjacent column swap
// of k,k+1 (0-indexed) or a symmetric adjacent permutation of k,k+1 (0-indexed)
#define FREE_WORKSPACE                          \
    SLIP_matrix_free(&T, option);               \
    SLIP_matrix_free(&A, option);               \
    SLIP_matrix_free(&L, option);               \
    SLIP_matrix_free(&U, option);               \
    SLIP_matrix_free(&rhos, option);            \
    SLIP_FREE(pinv);                            \
    SLIP_LU_analysis_free(&S, option);          \
    SLIP_FREE(option);                          \
    SLIP_finalize();

// print a CSC/MPZ SLIP_matrix as a dense grid of integers.
static void print_grid_csc_mpz (const char *name, const SLIP_matrix *M)
{
    int64_t m = M->m, n = M->n;
    mpz_t **G = (mpz_t **) malloc (m * sizeof (mpz_t *));
    for (int64_t i = 0; i < m; i++)
    {
        G[i] = (mpz_t *) malloc (n * sizeof (mpz_t));
        for (int64_t j = 0; j < n; j++) mpz_init (G[i][j]);
    }
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
        {
            mpz_set (G[M->i[p]][j], M->x.mpz[p]);
        }
    }

    // compute width for alignment
    int *w = (int *) calloc (n, sizeof (int));
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            int len = (int) mpz_sizeinbase (G[i][j], 10)
                    + (mpz_sgn (G[i][j]) < 0 ? 1 : 0);
            if (len > w[j]) w[j] = len;
        }
        if (w[j] < 1) w[j] = 1;
    }

    printf ("\n=== %s (%" PRId64 " x %" PRId64 ") ===\n", name, m, n);
    for (int64_t i = 0; i < m; i++)
    {
        printf ("  [");
        for (int64_t j = 0; j < n; j++)
        {
            gmp_printf (" %*Zd", w[j], G[i][j]);
        }
        printf (" ]\n");
    }

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++) mpz_clear (G[i][j]);
        free (G[i]);
    }
    free (G);
    free (w);
}

// build a SLIP CSC/MPZ matrix from a dense int array.
static SLIP_info dense_int_to_csc_mpz (SLIP_matrix **A_handle,
    const int64_t *dense, int64_t m, int64_t n,
    const SLIP_options *option)
{
    *A_handle = NULL;
    SLIP_info info;

    int64_t nnz = 0;
    for (int64_t k = 0; k < m * n; k++) if (dense[k] != 0) nnz++;
    if (nnz == 0) nnz = 1;   // SLIP requires nzmax >= 1

    SLIP_matrix *T = NULL;
    info = SLIP_matrix_allocate (&T, SLIP_TRIPLET, SLIP_FP64, m, n, nnz,
        false, true, option);
    if (info != SLIP_OK) return info;

    int64_t p = 0;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            int64_t v = dense[i * n + j];
            if (v != 0)
            {
                T->i[p] = i;
                T->j[p] = j;
                T->x.fp64[p] = (double) v;
                p++;
            }
        }
    }
    T->nz = p;

    info = SLIP_matrix_copy (A_handle, SLIP_CSC, SLIP_MPZ, T, option);
    SLIP_matrix_free (&T, option);
    return info;
}

// build the frame matrix F and then print it
static void print_frame_LU (const SLIP_matrix *L, const SLIP_matrix *U)
{
    int64_t n = L->n;
    mpz_t **F = (mpz_t **) malloc (n * sizeof (mpz_t *));
    for (int64_t i = 0; i < n; i++)
    {
        F[i] = (mpz_t *) malloc (n * sizeof (mpz_t));
        for (int64_t j = 0; j < n; j++) mpz_init (F[i][j]);
    }

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = L->p[j]; p < L->p[j+1]; p++)
        {
            int64_t i = L->i[p];
            if (i >= j) mpz_set (F[i][j], L->x.mpz[p]);
        }
    }

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = U->p[j]; p < U->p[j+1]; p++)
        {
            int64_t i = U->i[p];
            if (i < j) mpz_set (F[i][j], U->x.mpz[p]);
        }
    }

    int *w = (int *) calloc (n, sizeof (int));
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            int len = (int) mpz_sizeinbase (F[i][j], 10)
                    + (mpz_sgn (F[i][j]) < 0 ? 1 : 0);
            if (len > w[j]) w[j] = len;
        }
        if (w[j] < 1) w[j] = 1;
    }

    printf ("\n=== F = frame(L, U) (%" PRId64 " x %" PRId64 ") ===\n", n, n);
    for (int64_t i = 0; i < n; i++)
    {
        printf ("  [");
        for (int64_t j = 0; j < n; j++) gmp_printf (" %*Zd", w[j], F[i][j]);
        printf (" ]\n");
    }

    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < n; j++) mpz_clear (F[i][j]);
        free (F[i]);
    }
    free (F);
    free (w);
}

static void usage (const char *prog)
{
    fprintf (stderr,
        "usage: %s [-C <col> | -D <col>]\n", prog);
}

int main (int argc, char **argv)
{
    SLIP_initialize();

    SLIP_info ok;
    SLIP_matrix *T    = NULL;   // triplet form of F0
    SLIP_matrix *A    = NULL;   // CSC/MPZ form of F0
    SLIP_matrix *L    = NULL;
    SLIP_matrix *U    = NULL;
    SLIP_matrix *rhos = NULL;
    int64_t *pinv     = NULL;
    SLIP_LU_analysis *S = NULL;

    SLIP_options *option = SLIP_create_default_options();
    if (!option)
    {
        fprintf(stderr, "Error! OUT of MEMORY!\n");
        SLIP_finalize();
        return 1;
    }

    const int64_t n = 6;
    int64_t Adense[6][6] = {
        {3,0,2,0,0,0},
        {0,-6,30,0,-2,0},
        {0,2,-8,4,0,0},
        {0,0,0,1,0,0},
        {-8,0,0,0,-7,0},
        {2,0,0,0,0,-8},
    };

    enum { SWAP_NONE, SWAP_C, SWAP_D } mode = SWAP_NONE;
    int64_t col = -1;
    if (argc == 1)
    {
        // no args, normal
    }
    else if (argc == 3 && (strcmp (argv[1], "-C") == 0
                       ||  strcmp (argv[1], "-D") == 0))
    {
        mode = (argv[1][1] == 'C') ? SWAP_C : SWAP_D;
        col  = (int64_t) atoll (argv[2]);
        if (col < 0 || col >= n - 1)
        {
            fprintf (stderr, "col must be in [0, %" PRId64 "]\n", n - 2);
            usage (argv[0]);
            return 1;
        }
    }
    else
    {
        usage (argv[0]);
        return 1;
    }

    if (mode == SWAP_C || mode == SWAP_D)
    {
        for (int64_t i = 0; i < n; i++)
        {
            int64_t t = Adense[i][col];
            Adense[i][col]   = Adense[i][col + 1];
            Adense[i][col+1] = t;
        }
    }
    if (mode == SWAP_D)
    {
        for (int64_t j = 0; j < n; j++)
        {
            int64_t t = Adense[col][j];
            Adense[col][j]   = Adense[col + 1][j];
            Adense[col+1][j] = t;
        }
    }

    option->order = SLIP_NO_ORDERING;
    option->pivot = SLIP_DIAGONAL;
    OK(dense_int_to_csc_mpz (&A, (const int64_t *) Adense, n, n, option));

    OK(SLIP_LU_analyze(&S, A, option));
    OK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));

    //--------------------------------------------------------------------------
    // Print results
    //--------------------------------------------------------------------------

    print_grid_csc_mpz ("A", A);
    print_grid_csc_mpz ("L", L);
    print_grid_csc_mpz ("U", U);
    print_frame_LU (L, U);

    printf ("\n=== rhos (sequence of pivots) ===\n  [");
    for (int64_t i = 0; i < n; i++) gmp_printf (" %Zd", rhos->x.mpz[i]);
    printf (" ]\n");

    printf("\n=== pinv (inverse row permutation) ===\n");
    for (int64_t i = 0; i < n; i++)
    {
        printf("  pinv[%" PRId64 "] = %" PRId64 "\n", i, pinv[i]);
    }

    printf("\n=== q (column permutation from S) ===\n");
    for (int64_t j = 0; j < n; j++)
    {
        printf("  q[%" PRId64 "] = %" PRId64 "\n", j, S->q[j]);
    }

    FREE_WORKSPACE;
    return 0;
}
