//------------------------------------------------------------------------------
// SLIP_LU/Demo/ordering_ab.c: compare column orderings on one basis
//------------------------------------------------------------------------------
//
// Reads a qsx basis (same loader as qsx_basis), factorizes it three times
// (SLIP_COLAMD, SLIP_AMD, SLIP_NO_ORDERING), and reports for each:
//   analyze time, factor time, nnz(L)+nnz(U), and final pivot bit-length.
//
// Usage: ordering_ab <basis_file>

#include "demos.h"
#include <gmp.h>
#include <string.h>
#include <time.h>

#define FREE_WORKSPACE                  \
{                                       \
    SLIP_matrix_free(&A_trip, option);  \
    SLIP_matrix_free(&A,      option);  \
    SLIP_matrix_free(&L,      option);  \
    SLIP_matrix_free(&U,      option);  \
    SLIP_matrix_free(&rhos,   option);  \
    SLIP_LU_analysis_free(&S, option);  \
    SLIP_FREE(pinv);                    \
    SLIP_FREE(option);                  \
    SLIP_finalize();                    \
}

static SLIP_info read_basis_mpq(SLIP_matrix **A_handle, FILE *file,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *A = NULL;
    *A_handle = NULL;
    int64_t m, n, nnz;
    if (fscanf(file, "%"PRId64" %"PRId64" %"PRId64, &m, &n, &nnz) != 3)
        return SLIP_INCORRECT_INPUT;
    if (m != n) return SLIP_INCORRECT_INPUT;

    mpz_t *Lrow = (mpz_t *) SLIP_calloc((size_t) m, sizeof(mpz_t));
    if (!Lrow) return SLIP_OUT_OF_MEMORY;
    for (int64_t i = 0; i < m; i++) mpz_init(Lrow[i]);
    for (int64_t i = 0; i < m; i++)
    {
        if (gmp_fscanf(file, "%Zd", Lrow[i]) != 1)
        {
            for (int64_t k = 0; k < m; k++) mpz_clear(Lrow[k]);
            SLIP_FREE(Lrow);
            return SLIP_INCORRECT_INPUT;
        }
    }
    info = SLIP_matrix_allocate(&A, SLIP_TRIPLET, SLIP_MPQ, m, n, nnz,
        false, true, option);
    if (info != SLIP_OK)
    {
        for (int64_t k = 0; k < m; k++) mpz_clear(Lrow[k]);
        SLIP_FREE(Lrow);
        return info;
    }
    mpz_t num; mpz_init(num);
    for (int64_t k = 0; k < nnz; k++)
    {
        int64_t row, col;
        if (gmp_fscanf(file, "%"PRId64" %"PRId64" %Zd", &row, &col, num) != 3
            || row < 0 || row >= m || col < 0 || col >= n)
        {
            mpz_clear(num);
            for (int64_t i = 0; i < m; i++) mpz_clear(Lrow[i]);
            SLIP_FREE(Lrow);
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
        A->i[k] = row;
        A->j[k] = col;
        mpq_set_num(A->x.mpq[k], num);
        mpq_set_den(A->x.mpq[k], Lrow[row]);
        mpq_canonicalize(A->x.mpq[k]);
    }
    A->nz = nnz;
    mpz_clear(num);
    for (int64_t i = 0; i < m; i++) mpz_clear(Lrow[i]);
    SLIP_FREE(Lrow);
    *A_handle = A;
    return SLIP_OK;
}

static const char *order_name(SLIP_col_order o)
{
    switch (o)
    {
        case SLIP_NO_ORDERING: return "NONE";
        case SLIP_COLAMD:      return "COLAMD";
        case SLIP_AMD:         return "AMD";
    }
    return "?";
}

int main(int argc, char **argv)
{
    SLIP_initialize();
    SLIP_options *option = SLIP_create_default_options();
    SLIP_matrix *A_trip = NULL, *A = NULL;
    SLIP_matrix *L = NULL, *U = NULL, *rhos = NULL;
    SLIP_LU_analysis *S = NULL;
    int64_t *pinv = NULL;

    if (!option || argc < 2)
    {
        fprintf(stderr, "usage: ordering_ab <basis_file>\n");
        FREE_WORKSPACE;
        return 1;
    }

    SLIP_info ok;
    FILE *mf = fopen(argv[1], "r");
    if (!mf) { perror(argv[1]); FREE_WORKSPACE; return 1; }
    OK(read_basis_mpq(&A_trip, mf, option));
    fclose(mf);
    OK(SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ, A_trip, option));
    SLIP_matrix_free(&A_trip, option);
    printf("ordering_ab: B is %"PRId64"x%"PRId64", nnz=%"PRId64"\n",
        A->m, A->n, A->p[A->n]);

    SLIP_col_order orders[3] = { SLIP_COLAMD, SLIP_AMD, SLIP_NO_ORDERING };
    printf("\n%-8s %-10s %-10s %-12s %-12s %-12s\n",
        "order", "analyze", "factor", "nnz(L)", "nnz(U)", "det_bits");

    for (int oi = 0; oi < 3; oi++)
    {
        option->order = orders[oi];
        SLIP_LU_analysis_free(&S, option);
        SLIP_matrix_free(&L, option);
        SLIP_matrix_free(&U, option);
        SLIP_matrix_free(&rhos, option);
        SLIP_FREE(pinv);

        clock_t a0 = clock();
        OK(SLIP_LU_analyze(&S, A, option));
        clock_t a1 = clock();
        OK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));
        clock_t a2 = clock();

        size_t det_bits = 0;
        SLIP_mpz_sizeinbase(&det_bits, rhos->x.mpz[rhos->m - 1], 2);

        printf("%-8s %-10.3f %-10.3f %-12"PRId64" %-12"PRId64" %-12zu\n",
            order_name(orders[oi]),
            (double)(a1 - a0) / CLOCKS_PER_SEC,
            (double)(a2 - a1) / CLOCKS_PER_SEC,
            L->p[L->n], U->p[U->n], det_bits);
        fflush(stdout);
    }

    FREE_WORKSPACE;
    return 0;
}
