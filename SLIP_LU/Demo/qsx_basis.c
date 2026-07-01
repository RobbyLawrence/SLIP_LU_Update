//------------------------------------------------------------------------------------
// SLIP_LU/Demo/qsx_basis.c: demo to factorize a qsopt-ex basis snapshot with SLIP LU
//------------------------------------------------------------------------------------

// Reads a basis matrix B and right-hand-side b emitted by qsopt-ex'
// and converts them into the formats SLIP_LU expects (CSC SLIP_MPZ for B, dense SLIP_MPZ for b)
// and solves Bx=b exactly.
//
// Writes L, U, and x to ./sol_<problem>, condensing numbers with many zeros.
//
// Basis file layout (esolver.c:391):
//   nrows ncols nnz                       (ncols == nrows for a basis)
//   L_0 ... L_{nrows-1}                   one per line, row-wise denom LCM
//   row col int_val                       repeated nnz times, 0-indexed,
//                                         true entry B[row,col] = int_val/L_row
//
// RHS file layout (esolver.c:365):
//   n
//   denom num                             repeated n times; b[i] = num/denom
//
// Output x file (same shape as RHS):
//   n
//   denom num                             repeated n times
//
// Usage:
//   qsx_basis [-f FMT] <matrix_file> <rhs_file>
//
// FMT selects the input format (default: qsx):
//   qsx  -- qsopt-ex format described above (row-LCM matrix, denom/num RHS)
//   coo
//   csc

#include "demos.h"
#include <gmp.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#define FREE_WORKSPACE                  \
    SLIP_matrix_free(&A_trip, option);  \
    SLIP_matrix_free(&A, option);       \
    SLIP_matrix_free(&b_mpq, option);   \
    SLIP_matrix_free(&b, option);       \
    SLIP_matrix_free(&L, option);       \
    SLIP_matrix_free(&U, option);       \
    SLIP_matrix_free(&rhos, option);    \
    SLIP_LU_analysis_free(&S, option);  \
    SLIP_FREE(pinv);                    \
    SLIP_matrix_free(&x, option);       \
    SLIP_FREE(option);                  \
    SLIP_finalize();

static SLIP_info read_basis_mpq(SLIP_matrix **A_handle, FILE *file,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *A = NULL;
    *A_handle = NULL;

    int64_t m, n, nnz;
    if (fscanf(file, "%"PRId64" %"PRId64" %"PRId64, &m, &n, &nnz) != 3)
    {
        fprintf(stderr, "qsx_basis: bad basis header\n");
        return SLIP_INCORRECT_INPUT;
    }
    if (m != n)
    {
        fprintf(stderr, "qsx_basis: expected square basis, got %"PRId64
            " x %"PRId64"\n", m, n);
        return SLIP_INCORRECT_INPUT;
    }

    mpz_t *Lrow = (mpz_t *) SLIP_calloc((size_t) m, sizeof(mpz_t));
    if (!Lrow) return SLIP_OUT_OF_MEMORY;
    for (int64_t i = 0; i < m; i++) mpz_init(Lrow[i]);
    for (int64_t i = 0; i < m; i++)
    {
        if (gmp_fscanf(file, "%Zd", Lrow[i]) != 1)
        {
            fprintf(stderr, "qsx_basis: bad L[%"PRId64"]\n", i);
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

    mpz_t num;
    mpz_init(num);
    for (int64_t k = 0; k < nnz; k++)
    {
        int64_t row, col;
        if (gmp_fscanf(file, "%"PRId64" %"PRId64" %Zd",
                &row, &col, num) != 3)
        {
            fprintf(stderr, "qsx_basis: bad triplet at k=%"PRId64"\n", k);
            mpz_clear(num);
            for (int64_t i = 0; i < m; i++) mpz_clear(Lrow[i]);
            SLIP_FREE(Lrow);
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
        if (row < 0 || row >= m || col < 0 || col >= n)
        {
            fprintf(stderr, "qsx_basis: index out of range at k=%"PRId64"\n", k);
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
    if (fscanf(file, "%"PRId64, &n) != 1)
    {
        fprintf(stderr, "qsx_basis: bad rhs header\n");
        return SLIP_INCORRECT_INPUT;
    }

    info = SLIP_matrix_allocate(&b, SLIP_DENSE, SLIP_MPQ, n, 1, n,
        false, true, option);
    if (info != SLIP_OK) return info;

    mpz_t den, num;
    mpz_init(den);
    mpz_init(num);
    for (int64_t i = 0; i < n; i++)
    {
        if (gmp_fscanf(file, "%Zd %Zd", den, num) != 2)
        {
            fprintf(stderr, "qsx_basis: bad rhs entry %"PRId64"\n", i);
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

// Read a whitespace-separated integer CSC matrix directly into SLIP_CSC MPZ.
//   header:        m n nnz
//   col pointers:  n+1 integers, last one is nnz
//   row indices:   nnz integers (0-based)
//   values:        nnz integers
static SLIP_info read_csc_mpz(SLIP_matrix **A_handle, FILE *file,
    SLIP_options *option)
{
    SLIP_info info;
    SLIP_matrix *A = NULL;
    *A_handle = NULL;

    int64_t m, n, nnz;
    if (fscanf(file, "%"PRId64" %"PRId64" %"PRId64, &m, &n, &nnz) != 3)
    {
        fprintf(stderr, "qsx_basis: bad CSC header\n");
        return SLIP_INCORRECT_INPUT;
    }

    info = SLIP_matrix_allocate(&A, SLIP_CSC, SLIP_MPZ, m, n, nnz,
        false, true, option);
    if (info != SLIP_OK) return info;

    for (int64_t j = 0; j <= n; j++)
    {
        if (fscanf(file, "%"PRId64, &A->p[j]) != 1)
        {
            fprintf(stderr, "qsx_basis: bad CSC col-pointer %"PRId64"\n", j);
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
    }
    if (A->p[n] != nnz)
    {
        fprintf(stderr, "qsx_basis: CSC p[n]=%"PRId64" != nnz=%"PRId64"\n",
            A->p[n], nnz);
        SLIP_matrix_free(&A, option);
        return SLIP_INCORRECT_INPUT;
    }
    for (int64_t k = 0; k < nnz; k++)
    {
        if (fscanf(file, "%"PRId64, &A->i[k]) != 1
            || A->i[k] < 0 || A->i[k] >= m)
        {
            fprintf(stderr, "qsx_basis: bad CSC row index at k=%"PRId64"\n", k);
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
    }
    for (int64_t k = 0; k < nnz; k++)
    {
        if (gmp_fscanf(file, "%Zd", A->x.mpz[k]) != 1)
        {
            fprintf(stderr, "qsx_basis: bad CSC value at k=%"PRId64"\n", k);
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
    }

    *A_handle = A;
    return SLIP_OK;
}

// Print an mpz value to a file using a trailing-zero-compressed text form.
// Output is "<mantissa>e<count>" when the decimal representation has >=2
// trailing zeros, otherwise the plain decimal form.  e.g.
//   100000        -> "1e5"
//   -12000        -> "-12e3"
//   42            -> "42"
//   0             -> "0"
//   -10           -> "-10"     (only one trailing zero; not worth compressing)
// The `e` is unambiguous: GMP's %Zd never emits one.
static void fprint_mpz_tz(FILE *f, const mpz_t v)
{
    char *s = mpz_get_str(NULL, 10, v);
    size_t len = strlen(s);
    // Find start of magnitude (skip leading '-').
    size_t mag_start = (s[0] == '-') ? 1 : 0;
    size_t mag_len = len - mag_start;
    // "0" stays "0".
    if (mag_len == 1 && s[mag_start] == '0')
    {
        fputs(s, f);
        free(s);
        return;
    }
    // Count trailing zeros.
    size_t tz = 0;
    while (tz < mag_len - 1 && s[len - 1 - tz] == '0') tz++;
    if (tz < 2)
    {
        fputs(s, f);
    }
    else
    {
        // Truncate the trailing zeros and emit "<mantissa>e<tz>".
        s[len - tz] = '\0';
        fprintf(f, "%se%zu", s, tz);
    }
    free(s);
}

// Write a CSC SLIP_MPZ matrix as a 1-based triplet file.  Values use the
// trailing-zero-compressed form above; explicit-zero entries are skipped.
// Header `nnz` reflects the number of stored entries.
static int write_csc_mpz_triplet(const char *path, const SLIP_matrix *M)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 1; }

    // First pass: count nonzero entries.
    int64_t nnz_stored = 0;
    for (int64_t j = 0; j < M->n; j++)
    {
        for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
        {
            if (mpz_sgn(M->x.mpz[p]) != 0) nnz_stored++;
        }
    }

    fprintf(f, "%"PRId64" %"PRId64" %"PRId64"\n", M->m, M->n, nnz_stored);
    for (int64_t j = 0; j < M->n; j++)
    {
        for (int64_t p = M->p[j]; p < M->p[j+1]; p++)
        {
            if (mpz_sgn(M->x.mpz[p]) == 0) continue;
            fprintf(f, "%"PRId64" %"PRId64" ", M->i[p] + 1, j + 1);
            fprint_mpz_tz(f, M->x.mpz[p]);
            fputc('\n', f);
        }
    }
    fclose(f);
    return 0;
}

// Write a dense SLIP_MPQ vector in qsopt-ex's "denom num" format.
static int write_dense_mpq(const char *path, const SLIP_matrix *v)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    int64_t n = v->m * v->n;
    fprintf(f, "%"PRId64"\n", n);
    for (int64_t i = 0; i < n; i++)
    {
        gmp_fprintf(f, "%Zd %Zd\n",
            mpq_denref(v->x.mpq[i]), mpq_numref(v->x.mpq[i]));
    }
    fclose(f);
    return 0;
}

// Derive "<problem>" and "kN" tag from the basis-file path. Expects something
// matching ".../<problem>_Bases/basis_kN_B.txt".  Falls back to "unknown"/""
// if the pattern doesn't fit.
static void derive_names(const char *basis_path, char *prob, size_t prob_sz,
    char *ktag, size_t ktag_sz)
{
    snprintf(prob, prob_sz, "unknown");
    snprintf(ktag, ktag_sz, " ");

    // parent directory: everything before the last '/'
    const char *slash = strrchr(basis_path, '/');
    const char *base = slash ? slash + 1 : basis_path;

    // problem name = parent dir basename minus "_Bases" suffix
    if (slash)
    {
        const char *parent_end = slash;
        const char *parent_start = basis_path;
        for (const char *p = slash - 1; p >= basis_path; p--)
        {
            if (*p == '/') { parent_start = p + 1; break; }
        }
        size_t plen = (size_t)(parent_end - parent_start);
        const char *suf = "_Bases";
        size_t slen = strlen(suf);
        if (plen > slen && strncmp(parent_end - slen, suf, slen) == 0)
        {
            plen -= slen;
        }
        if (plen >= prob_sz) plen = prob_sz - 1;
        memcpy(prob, parent_start, plen);
        prob[plen] = '\0';
    }

    // k-tag = "kN" extracted from basename "basis_kN_B.txt"
    const char *k = strstr(base, "_k");
    if (k)
    {
        k += 1; // points at 'k'
        const char *end = strchr(k, '_');
        if (!end) end = k + strlen(k);
        size_t klen = (size_t)(end - k);
        if (klen >= ktag_sz) klen = ktag_sz - 1;
        memcpy(ktag, k, klen);
        ktag[klen] = '\0';
    }
}

int main(int argc, char **argv)
{
    SLIP_initialize();

    SLIP_info ok;
    SLIP_matrix *A_trip = NULL;     // triplet MPQ
    SLIP_matrix *A      = NULL;     // CSC MPZ
    SLIP_matrix *b_mpq  = NULL;     // dense MPQ
    SLIP_matrix *b      = NULL;     // dense MPZ
    SLIP_matrix *L      = NULL;
    SLIP_matrix *U      = NULL;
    SLIP_matrix *rhos   = NULL;
    SLIP_LU_analysis *S = NULL;
    int64_t *pinv       = NULL;
    SLIP_matrix *x      = NULL;
    SLIP_options *option = SLIP_create_default_options();
    if (!option)
    {
        fprintf(stderr, "qsx_basis: out of memory\n");
        SLIP_finalize();
        return 1;
    }

    enum { FMT_QSX = 0, FMT_COO = 1, FMT_CSC = 2 } fmt = FMT_QSX;
    const char *basis_name = NULL;
    const char *rhs_name   = NULL;

    // Parse optional "-f FMT" then two positional args.
    int ai = 1;
    if (ai < argc && strcmp(argv[ai], "-f") == 0)
    {
        if (ai + 1 >= argc)
        {
            fprintf(stderr, "qsx_basis: -f requires an argument (qsx|coo|csc)\n");
            FREE_WORKSPACE;
            return 1;
        }
        const char *v = argv[ai + 1];
        if      (strcmp(v, "qsx") == 0) fmt = FMT_QSX;
        else if (strcmp(v, "coo") == 0) fmt = FMT_COO;
        else if (strcmp(v, "csc") == 0) fmt = FMT_CSC;
        else
        {
            fprintf(stderr, "qsx_basis: unknown format '%s' (use qsx|coo|csc)\n",
                v);
            FREE_WORKSPACE;
            return 1;
        }
        ai += 2;
    }
    int remaining = argc - ai;
    if (remaining != 2)
    {
        fprintf(stderr,
            "usage: qsx_basis [-f qsx|coo|csc] <matrix_file> <rhs_file>\n");
        FREE_WORKSPACE;
        return 1;
    }
    basis_name = argv[ai];
    rhs_name   = argv[ai + 1];
    const char *fmt_name = (fmt == FMT_QSX) ? "qsx"
                         : (fmt == FMT_COO) ? "coo" : "csc";
    printf("qsx_basis: fmt=%s\n             matrix=%s\n             rhs=%s\n",
        fmt_name, basis_name, rhs_name);

    char prob[256], ktag[64];
    derive_names(basis_name, prob, sizeof prob, ktag, sizeof ktag);
    char outdir[512];
    snprintf(outdir, sizeof outdir, "sol_%s", prob);
    if (mkdir(outdir, 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "qsx_basis: cannot mkdir %s: %s\n",
            outdir, strerror(errno));
        FREE_WORKSPACE;
        return 1;
    }
    printf("qsx_basis: outdir=%s/  tag=%s\n", outdir,
        ktag[0] ? ktag : "(none)");

    FILE *mf = fopen(basis_name, "r");
    if (!mf) { perror(basis_name); FREE_WORKSPACE; return 1; }
    if (fmt == FMT_QSX)
    {
        OK(read_basis_mpq(&A_trip, mf, option));
    }
    else if (fmt == FMT_COO)
    {
        OK(SLIP_tripread(&A, mf, option));     // already CSC MPZ
    }
    else
    {
        OK(read_csc_mpz(&A, mf, option));      // already CSC MPZ
    }
    fclose(mf);

    FILE *rf = fopen(rhs_name, "r");
    if (!rf) { perror(rhs_name); FREE_WORKSPACE; return 1; }
    if (fmt == FMT_QSX)
    {
        OK(read_rhs_mpq(&b_mpq, rf, option));
    }
    else
    {
        OK(SLIP_read_dense(&b, rf, option));   // already DENSE MPZ
    }
    fclose(rf);

    // Convert qsx MPQ inputs to MPZ.  coo/csc inputs are already MPZ.
    if (fmt == FMT_QSX)
    {
        if (A_trip->m != b_mpq->m)
        {
            fprintf(stderr, "qsx_basis: size mismatch (B is %"PRId64"x%"PRId64
                ", b has %"PRId64")\n", A_trip->m, A_trip->n, b_mpq->m);
            FREE_WORKSPACE;
            return 1;
        }
        OK(SLIP_matrix_copy(&A, SLIP_CSC,   SLIP_MPZ, A_trip, option));
        OK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ, b_mpq,  option));
        SLIP_matrix_free(&A_trip, option);
        SLIP_matrix_free(&b_mpq,  option);
    }
    else if (A->n != b->m)
    {
        fprintf(stderr, "qsx_basis: size mismatch (A is %"PRId64"x%"PRId64
            ", b has %"PRId64")\n", A->m, A->n, b->m);
        FREE_WORKSPACE;
        return 1;
    }

    printf("qsx_basis: B is %"PRId64"x%"PRId64", nnz=%"PRId64"\n",
        A->m, A->n, A->p[A->n]);

    // Expert pipeline so we can capture L and U.
    clock_t t0 = clock();
    OK(SLIP_LU_analyze(&S, A, option));
    clock_t t1 = clock();
    OK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));
    clock_t t2 = clock();
    option->check = true;
    OK(SLIP_LU_solve(&x, b, A, L, U, rhos, S, pinv, option));
    clock_t t3 = clock();

    printf("qsx_basis: analyze=%lfs factor=%lfs solve=%lfs\n",
        (double)(t1-t0)/CLOCKS_PER_SEC,
        (double)(t2-t1)/CLOCKS_PER_SEC,
        (double)(t3-t2)/CLOCKS_PER_SEC);
    // Bit-length of the final pivot (= |det(B)|) and max entry bit-length
    // in L and U.  The pivot bit-length is the Hadamard-bound cost driver:
    // every mpz_mul/mpz_divexact in the late columns operates on integers
    // this large.
    size_t det_bits = 0;
    SLIP_mpz_sizeinbase(&det_bits, rhos->x.mpz[rhos->m - 1], 2);
    size_t max_L_bits = 0, max_U_bits = 0;
    for (int64_t i = 0; i < L->p[L->n]; i++)
    {
        size_t b = 0; SLIP_mpz_sizeinbase(&b, L->x.mpz[i], 2);
        if (b > max_L_bits) max_L_bits = b;
    }
    for (int64_t i = 0; i < U->p[U->n]; i++)
    {
        size_t b = 0; SLIP_mpz_sizeinbase(&b, U->x.mpz[i], 2);
        if (b > max_U_bits) max_U_bits = b;
    }
    printf("qsx_basis: nnz(L)=%"PRId64" nnz(U)=%"PRId64
        " det_bits=%zu max_L_bits=%zu max_U_bits=%zu\n",
        L->p[L->n], U->p[U->n], det_bits, max_L_bits, max_U_bits);

    // Write L, U, x.
    char path[1024];
    const char *tag = ktag[0] ? ktag : "out";
    snprintf(path, sizeof path, "%s/L_%s.txt", outdir, tag);
    if (write_csc_mpz_triplet(path, L)) { FREE_WORKSPACE; return 1; }
    printf("qsx_basis: wrote %s\n", path);

    snprintf(path, sizeof path, "%s/U_%s.txt", outdir, tag);
    if (write_csc_mpz_triplet(path, U)) { FREE_WORKSPACE; return 1; }
    printf("qsx_basis: wrote %s\n", path);

    snprintf(path, sizeof path, "%s/x_%s.txt", outdir, tag);
    if (write_dense_mpq(path, x)) { FREE_WORKSPACE; return 1; }
    printf("qsx_basis: wrote %s\n", path);

    FREE_WORKSPACE;
    printf("qsx_basis: done\n");
    return 0;
}
