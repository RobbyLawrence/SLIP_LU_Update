#-------------------------------------------------------------------------------
# SLIP_LU/Demo/Makefile: compile and run the demos
#-------------------------------------------------------------------------------

# SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
# Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
# SLIP_LU/License for the license.

#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

UNAME_S := $(shell uname -s)
ifneq (,(findstring CYGWIN,$(UNAME_S)))
	export PATH := ../../lib:$(PATH)
endif

# uncomment for extra error checking:
CFLAGS += -Wall -Wextra -Wpedantic -Werror

# CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -O
I = -I../Include -I../Source -I../../SuiteSparse_config -I../../COLAMD/Include -I../../AMD/Include

# LDFLAGS = -L../../lib

LDLIBS += -lm -lgmp -lmpfr -lcolamd -lamd -lsliplu
CS = $(LDLIBS)


# Build all demo programs.  Nothing is run automatically; invoke a program
# (e.g. ./perm) by hand once it is built.
all: example example2 SLIPLU qsx_basis update_demo perm

lib:
	( cd ../Lib ; $(MAKE) )

# Compile a source file into an object file.  Only sources whose .c (or the
# headers/Makefile they depend on) changed are recompiled.
%.o: %.c Makefile
	$(CC) $(CF) $(I) -c $< -o $@

# Header dependencies: recompile the affected objects when a header changes.
demos.o example2.o SLIPLU.o qsx_basis.o update_demo.o perm.o ordering_ab.o: demos.h
slip_lu_update.o update_demo.o: slip_lu_update.h

# Link the demo programs from their objects.  'lib' is an order-only
# prerequisite (after the '|') so the library is kept up to date without
# forcing the programs to relink on every invocation.
example: example.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ example.o $(CS)

example2: example2.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ example2.o demos.o $(CS)

SLIPLU: SLIPLU.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ SLIPLU.o demos.o $(CS)

qsx_basis: qsx_basis.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ qsx_basis.o demos.o $(CS)

update_demo: update_demo.o slip_lu_update.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ update_demo.o slip_lu_update.o demos.o $(CS)

perm: perm.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ perm.o demos.o $(CS)

ordering_ab: ordering_ab.o demos.o | lib
	$(CC) $(LDFLAGS) $(CF) -o $@ ordering_ab.o demos.o $(CS)

clean:
	- $(RM) *.o

purge: distclean

distclean: clean
	- $(RM) -r example example2 example3 example4 example5 SLIPLU qsx_basis update_demo perm ordering_ab *.a *.dSYM *.obj *.dll

.PHONY: default all lib clean purge distclean
