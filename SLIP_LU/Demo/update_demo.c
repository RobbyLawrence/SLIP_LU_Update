//------------------------------------------------------------------------------
// SLIP_LU/Demo/update_demo.c: benchmark column-replacement update vs refactor
//------------------------------------------------------------------------------

// Reads a basis matrix B (qsx format, same loader as qsx_basis), factorizes it,
// then for each requested replacement position p, builds B' by replacing column
// q[p] with (col q[p] + col q[r]) for a deterministic r != p (keeps B'
// nonsingular: det(B') = det(B)), and times:
//   (1) full SLIP_LU_factorize of B'
//   (2) slip_lu_update_column_replace on a clone of the original factors
// then solves B' x = b with the resulting factors via SLIP_LU_solve and checks
// the two solutions agree exactly.
//
// Usage: update_demo <basis_file> <rhs_file> [num_replacements]

#include "demos.h"
#include "slip_lu_update.h"
#include <gmp.h>
#include <string.h>
#include <time.h>

#define FREE_WORKSPACE                          \
{                                               \
    SLIP_matrix_free(&A_trip, option);          \
    SLIP_matrix_free(&A,      option);          \
    SLIP_matrix_free(&B_prime,option);          \
    SLIP_matrix_free(&b_mpq,  option);          \
    SLIP_matrix_free(&b,      option);          \
    SLIP_matrix_free(&L,      option);          \
    SLIP_matrix_free(&U,      option);          \
    SLIP_matrix_free(&rhos,   option);          \
    SLIP_matrix_free(&L_full, option);          \
    SLIP_matrix_free(&U_full, option);          \
    SLIP_matrix_free(&rhos_full, option);       \
    SLIP_matrix_free(&L_upd,  option);          \
    SLIP_matrix_free(&U_upd,  option);          \
    SLIP_matrix_free(&rhos_upd, option);        \
    SLIP_matrix_free(&x_full, option);          \
    SLIP_matrix_free(&x_upd,  option);          \
    SLIP_LU_analysis_free(&S, option);          \
    SLIP_FREE(pinv);                            \
    SLIP_FREE(pinv_full);                       \
    SLIP_FREE(pinv_upd);                        \
    SLIP_FREE(option);                          \
    SLIP_finalize();                            \
}

// qsx basis reader (lifted from qsx_basis.c, MPQ triplet output).
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

static SLIP_info read_rhs_mpq(SLIP_matrix **b_handle, FILE *file,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *b = NULL;
    *b_handle = NULL;
    int64_t n;
    if (fscanf(file, "%"PRId64, &n) != 1) return SLIP_INCORRECT_INPUT;
    info = SLIP_matrix_allocate(&b, SLIP_DENSE, SLIP_MPQ, n, 1, n,
        false, true, option);
    if (info != SLIP_OK) return info;
    mpz_t den, num; mpz_init(den); mpz_init(num);
    for (int64_t i = 0; i < n; i++)
    {
        if (gmp_fscanf(file, "%Zd %Zd", den, num) != 2)
        {
            mpz_clear(den); mpz_clear(num);
            SLIP_matrix_free(&b, option);
            return SLIP_INCORRECT_INPUT;
        }
        mpq_set_num(b->x.mpq[i], num);
        mpq_set_den(b->x.mpq[i], den);
        mpq_canonicalize(b->x.mpq[i]);
    }
    mpz_clear(den); mpz_clear(num);
    *b_handle = b;
    return SLIP_OK;
}

// Build B' = clone of A with column `col_replaced` overwritten by
// (original col_replaced + original col_donor). New nnz pattern is the union;
// values added.  B' is fresh CSC SLIP_MPZ.
static SLIP_info build_modified_matrix(SLIP_matrix **B_out, const SLIP_matrix *A,
    int64_t col_replaced, int64_t col_donor, SLIP_options *option)
{
    SLIP_info info;
    int64_t n = A->n;
    // Count nnz for new column
    int64_t cr_start = A->p[col_replaced], cr_end = A->p[col_replaced + 1];
    int64_t cd_start = A->p[col_donor],    cd_end = A->p[col_donor + 1];
    // Build dense scratch for the new column
    mpz_t *scratch = (mpz_t *) SLIP_calloc((size_t) n, sizeof(mpz_t));
    int8_t *touched = (int8_t *) SLIP_calloc((size_t) n, sizeof(int8_t));
    if (!scratch || !touched) { SLIP_FREE(scratch); SLIP_FREE(touched);
        return SLIP_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < n; i++) mpz_init(scratch[i]);
    for (int64_t p = cr_start; p < cr_end; p++)
    {
        mpz_add(scratch[A->i[p]], scratch[A->i[p]], A->x.mpz[p]);
        touched[A->i[p]] = 1;
    }
    for (int64_t p = cd_start; p < cd_end; p++)
    {
        mpz_add(scratch[A->i[p]], scratch[A->i[p]], A->x.mpz[p]);
        touched[A->i[p]] = 1;
    }
    int64_t new_col_nnz = 0;
    for (int64_t i = 0; i < n; i++)
    {
        if (touched[i] && mpz_sgn(scratch[i]) != 0) new_col_nnz++;
    }

    int64_t old_col_nnz = cr_end - cr_start;
    int64_t total_nnz = A->p[n] - old_col_nnz + new_col_nnz;

    SLIP_matrix *B = NULL;
    info = SLIP_matrix_allocate(&B, SLIP_CSC, SLIP_MPZ, n, n, total_nnz,
        false, true, option);
    if (info != SLIP_OK)
    {
        for (int64_t i = 0; i < n; i++) mpz_clear(scratch[i]);
        SLIP_FREE(scratch); SLIP_FREE(touched);
        return info;
    }

    int64_t out = 0;
    for (int64_t j = 0; j < n; j++)
    {
        B->p[j] = out;
        if (j == col_replaced)
        {
            for (int64_t i = 0; i < n; i++)
            {
                if (touched[i] && mpz_sgn(scratch[i]) != 0)
                {
                    B->i[out] = i;
                    mpz_set(B->x.mpz[out], scratch[i]);
                    out++;
                }
            }
        }
        else
        {
            for (int64_t p = A->p[j]; p < A->p[j+1]; p++)
            {
                B->i[out] = A->i[p];
                mpz_set(B->x.mpz[out], A->x.mpz[p]);
                out++;
            }
        }
    }
    B->p[n] = out;
    // Copy A's scale (mpq)
    mpq_set(B->scale, A->scale);

    for (int64_t i = 0; i < n; i++) mpz_clear(scratch[i]);
    SLIP_FREE(scratch); SLIP_FREE(touched);
    *B_out = B;
    return SLIP_OK;
}

// Deep-copy a CSC MPZ matrix.
static SLIP_info matrix_clone_csc_mpz(SLIP_matrix **out, const SLIP_matrix *A,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *C = NULL;
    int64_t n = A->n, nnz = A->p[A->n];
    info = SLIP_matrix_allocate(&C, SLIP_CSC, SLIP_MPZ, A->m, n, nnz,
        false, true, option);
    if (info != SLIP_OK) return info;
    for (int64_t j = 0; j <= n; j++) C->p[j] = A->p[j];
    for (int64_t k = 0; k < nnz; k++)
    {
        C->i[k] = A->i[k];
        mpz_set(C->x.mpz[k], A->x.mpz[k]);
    }
    mpq_set(C->scale, A->scale);
    *out = C;
    return SLIP_OK;
}

// Deep-copy a dense MPZ matrix (rhos).
static SLIP_info dense_clone_mpz(SLIP_matrix **out, const SLIP_matrix *src,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *C = NULL;
    int64_t m = src->m, n = src->n;
    info = SLIP_matrix_allocate(&C, SLIP_DENSE, SLIP_MPZ, m, n, m*n,
        false, true, option);
    if (info != SLIP_OK) return info;
    for (int64_t i = 0; i < m*n; i++) mpz_set(C->x.mpz[i], src->x.mpz[i]);
    mpq_set(C->scale, src->scale);
    *out = C;
    return SLIP_OK;
}

static int64_t mat_nnz(const SLIP_matrix *M) { return M->p[M->n]; }

int main(int argc, char **argv)
{
    SLIP_initialize();
    SLIP_options *option = SLIP_create_default_options();
    SLIP_matrix *A_trip = NULL, *A = NULL, *B_prime = NULL;
    SLIP_matrix *b_mpq = NULL, *b = NULL;
    SLIP_matrix *L = NULL, *U = NULL, *rhos = NULL;            // original
    SLIP_matrix *L_full = NULL, *U_full = NULL, *rhos_full = NULL; // refactor of B'
    SLIP_matrix *L_upd  = NULL, *U_upd  = NULL, *rhos_upd  = NULL; // updated
    SLIP_matrix *x_full = NULL, *x_upd = NULL;
    SLIP_LU_analysis *S = NULL;
    int64_t *pinv = NULL, *pinv_full = NULL, *pinv_upd = NULL;

    if (!option) { fprintf(stderr, "OOM\n"); SLIP_finalize(); return 1; }

    if (argc < 3)
    {
        fprintf(stderr,
            "usage: update_demo <basis_file> <rhs_file> [num_replacements]\n");
        FREE_WORKSPACE;
        return 1;
    }
    const char *basis_path = argv[1];
    const char *rhs_path   = argv[2];
    int num_reps = (argc >= 4) ? atoi(argv[3]) : 5;
    if (num_reps < 1) num_reps = 1;

    FILE *mf = fopen(basis_path, "r");
    if (!mf) { perror(basis_path); FREE_WORKSPACE; return 1; }
    SLIP_info ok;
    OK(read_basis_mpq(&A_trip, mf, option));
    fclose(mf);
    FILE *rf = fopen(rhs_path, "r");
    if (!rf) { perror(rhs_path); FREE_WORKSPACE; return 1; }
    OK(read_rhs_mpq(&b_mpq, rf, option));
    fclose(rf);

    OK(SLIP_matrix_copy(&A, SLIP_CSC,   SLIP_MPZ, A_trip, option));
    OK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ, b_mpq,  option));
    SLIP_matrix_free(&A_trip, option);
    SLIP_matrix_free(&b_mpq,  option);

    printf("update_demo: B is %"PRId64"x%"PRId64", nnz=%"PRId64"\n",
        A->m, A->n, A->p[A->n]);

    // ------------------------------------------------------------------
    // Original factorization (kept across all replacements as a base)
    // ------------------------------------------------------------------
    clock_t t0 = clock();
    OK(SLIP_LU_analyze(&S, A, option));
    clock_t t1 = clock();
    OK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));
    clock_t t2 = clock();
    printf("update_demo: baseline analyze=%.3fs factor=%.3fs nnz(L)=%"PRId64
        " nnz(U)=%"PRId64"\n",
        (double)(t1-t0)/CLOCKS_PER_SEC, (double)(t2-t1)/CLOCKS_PER_SEC,
        mat_nnz(L), mat_nnz(U));

    int64_t n = A->n;
    printf("\n%-4s %-6s %-6s %-12s %-12s %-8s %-12s %-12s\n",
        "rep", "p", "col", "refactor(s)", "update(s)", "speedup",
        "nnz(L)U_new", "match");

    for (int i = 0; i < num_reps; i++)
    {
        // pick a position p spread across [n/8, 7n/8]
        int64_t p_pos = (n * (1 + i)) / (num_reps + 1);
        if (p_pos < 0) p_pos = 0;
        if (p_pos >= n) p_pos = n - 1;
        int64_t col_replaced = S->q[p_pos];
        int64_t col_donor = S->q[(p_pos + 1) % n];
        if (col_donor == col_replaced) col_donor = S->q[(p_pos + 2) % n];

        SLIP_matrix_free(&B_prime, option);
        OK(build_modified_matrix(&B_prime, A, col_replaced, col_donor, option));

        // -- (1) full refactorize
        SLIP_matrix_free(&L_full, option);
        SLIP_matrix_free(&U_full, option);
        SLIP_matrix_free(&rhos_full, option);
        SLIP_FREE(pinv_full);
        clock_t r0 = clock();
        OK(SLIP_LU_factorize(&L_full, &U_full, &rhos_full, &pinv_full,
            B_prime, S, option));
        clock_t r1 = clock();
        double t_refactor = (double)(r1 - r0) / CLOCKS_PER_SEC;

        // -- (2) update: clone the baseline factors, then call update.
        SLIP_matrix_free(&L_upd, option);
        SLIP_matrix_free(&U_upd, option);
        SLIP_matrix_free(&rhos_upd, option);
        SLIP_FREE(pinv_upd);
        OK(matrix_clone_csc_mpz(&L_upd, L, option));
        OK(matrix_clone_csc_mpz(&U_upd, U, option));
        OK(dense_clone_mpz    (&rhos_upd, rhos, option));
        pinv_upd = (int64_t *) SLIP_malloc(n * sizeof(int64_t));
        if (!pinv_upd) { FREE_WORKSPACE; return 1; }
        for (int64_t k = 0; k < n; k++) pinv_upd[k] = pinv[k];

        clock_t u0 = clock();
        SLIP_info uinfo = slip_lu_update_column_replace(&L_upd, &U_upd,
            rhos_upd, pinv_upd, S, B_prime, col_replaced, option);
        clock_t u1 = clock();
        double t_update = (double)(u1 - u0) / CLOCKS_PER_SEC;
        if (uinfo != SLIP_OK)
        {
            printf("%-4d %-6"PRId64" %-6"PRId64" %-12.3f %-12.3f UPDATE FAIL "
                "(info=%d)\n",
                i, p_pos, col_replaced, t_refactor, t_update, (int) uinfo);
            continue;
        }

        // -- correctness: solve B' x = b with both and compare
        SLIP_matrix_free(&x_full, option);
        SLIP_matrix_free(&x_upd,  option);
        OK(SLIP_LU_solve(&x_full, b, B_prime, L_full, U_full, rhos_full,
            S, pinv_full, option));
        OK(SLIP_LU_solve(&x_upd,  b, B_prime, L_upd,  U_upd,  rhos_upd,
            S, pinv_upd, option));

        int match = 1, r;
        for (int64_t k = 0; k < n && match; k++)
        {
            mpq_t *a = &x_full->x.mpq[k];
            mpq_t *bq = &x_upd->x.mpq[k];
            if (mpq_cmp(*a, *bq) != 0) match = 0;
        }
        (void) r;

        double speedup = (t_update > 0) ? (t_refactor / t_update) : 0.0;
        printf("%-4d %-6"PRId64" %-6"PRId64" %-12.3f %-12.3f %-8.2f "
            "%-12"PRId64" %-12s\n",
            i, p_pos, col_replaced, t_refactor, t_update, speedup,
            mat_nnz(L_upd) + mat_nnz(U_upd),
            match ? "OK" : "MISMATCH");
        fflush(stdout);
    }

    FREE_WORKSPACE;
    return 0;
}
