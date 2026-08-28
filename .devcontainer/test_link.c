/* Minimal test that links against SLIP LU.
 * Initializes the library, allocates a small SLIP_options, frees it, and
 * finalizes. If this builds and returns 0, the toolchain and library layout
 * inside the devcontainer are working. */

#include <stdio.h>
#include "SLIP_LU.h"

int main(void)
{
    SLIP_info info = SLIP_initialize();
    if (info != SLIP_OK)
    {
        fprintf(stderr, "SLIP_initialize failed (%d)\n", info);
        return 1;
    }

    SLIP_options *option = SLIP_create_default_options();
    if (option == NULL)
    {
        fprintf(stderr, "SLIP_create_default_options failed\n");
        SLIP_finalize();
        return 1;
    }

    printf("SLIP LU link test OK (default tolerance = %g)\n", option->tol);

    SLIP_FREE(option);
    SLIP_finalize();
    return 0;
}
