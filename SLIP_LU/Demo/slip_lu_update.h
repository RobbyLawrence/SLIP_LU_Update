//------------------------------------------------------------------------------
// SLIP_LU/Demo/slip_lu_update.h: column-replacement update for SLIP LU
//------------------------------------------------------------------------------

#ifndef SLIP_LU_UPDATE_H
#define SLIP_LU_UPDATE_H

#include "SLIP_LU.h"

// Update the SLIP LU factorization (L, U, rhos, pinv) of B to that of B',
// where B' equals B with original column `col` replaced by `a_new`.  Q (column
// order) is reused.  On singular B', returns SLIP_SINGULAR.
//
// On entry, *L_handle and *U_handle are the factors produced by
// SLIP_LU_factorize (row indices already permuted by pinv at output).  On
// successful return, they are freed and replaced by new factors corresponding
// to B'.  rhos and pinv are reused storage and updated in place (rhos values
// rewritten via mpz_set; pinv reassigned).
//
// `A_new` must be the CSC SLIP_MPZ matrix with column `col` already replaced
// by a_new (the original A is not consulted for column `col`).
//
// Milestone A: recomputes every column k = p..n-1 unconditionally where
// p is the position with S->q[p] == col.

SLIP_info slip_lu_update_column_replace
(
    SLIP_matrix **L_handle,         // in/out: lower factor (rebuilt)
    SLIP_matrix **U_handle,         // in/out: upper factor (rebuilt)
    SLIP_matrix  *rhos,             // in/out: pivot sequence (n x 1)
    int64_t      *pinv,             // in/out: inverse row permutation (size n)
    const SLIP_LU_analysis *S,      // in: column permutation (reused)
    const SLIP_matrix *A_new,       // in: B with column `col` replaced
    int64_t       col,              // in: original column index that changed
    const SLIP_options *option
);

#endif
