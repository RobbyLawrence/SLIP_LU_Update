//------------------------------------------------------------------------------
// SLIP_LU/Demo/sdipps.h: sparse DIP push-and-swap update for SLIP LU
//------------------------------------------------------------------------------

// Port of sdipps.py to C, using mpz_t (via SLIP_gmp wrappers) throughout.
//
// SDIPPS takes an existing SLIP LU factorization (L, U, rhos, pinv) of some
// permuted basis and updates it in place to reflect a column replacement in
// the underlying matrix.  L and U are CSC MPZ (row indices in the stored form
// are pinv-permuted, matching SLIP_LU_factorize output).  Ftmp is accumulated
// internally as a COO triplet list, then split into new CSC L and U at the
// end.  The seed/history vectors mu, h are dense.
//
// The algorithmic answer sheet is sdipps.py; every branch here is a literal
// port of the corresponding Python code path.  Diagonal push is preferred;
// column push is the fallback when the diagonal push condition fails.

#ifndef SDIPPS_H
#define SDIPPS_H

#include "SLIP_LU.h"

// SDIPPS: run the sDIP push-and-swap chain from pivot position K to n-1.
//
// On entry:
//   *L_handle, *U_handle : CSC MPZ factors of the current basis
//                          (row indices pinv-permuted).
//   rhos                 : dense MPZ, size n, pivot sequence.
//   pinv                 : inverse row permutation, size n.
//   q                    : column permutation, size n.
//   A_new                : new basis (CSC MPZ) after the column replacement.
//                          Not consulted by SDIPPS itself; the caller supplies
//                          it for callers that want to keep it alongside.
//   K                    : starting pivot index; positions 0..K-1 in the
//                          current ordering are untouched.
//
// On successful return:
//   *L_handle, *U_handle : freed and replaced by new factors.
//   rhos                 : rewritten in place with the new pivot sequence.
//   pinv                 : composed with the SDIPPS row swaps (fP).
//   q                    : composed with the SDIPPS col swaps (fQ).
//
// The new L, U are output with row indices in pinv-permuted form to match
// SLIP_LU_factorize's convention, so they drop straight into SLIP_LU_solve.

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
);

#endif
