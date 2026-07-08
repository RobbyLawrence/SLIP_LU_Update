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

BIN = bin
OBJ = obj

$(BIN):
	mkdir -p $(BIN)

$(OBJ):
	mkdir -p $(OBJ)

# Build all demo programs.  Nothing is run automatically; invoke a program
# (e.g. ./bin/perm) by hand once it is built.
all: $(BIN)/example $(BIN)/example2 $(BIN)/SLIPLU $(BIN)/qsx_basis $(BIN)/update_demo $(BIN)/perm

lib:
	( cd ../Lib ; $(MAKE) )

# Compile a source file into an object file.  Only sources whose .c (or the
# headers/Makefile they depend on) changed are recompiled.
$(OBJ)/%.o: %.c Makefile | $(OBJ)
	$(CC) $(CF) $(I) -c $< -o $@

# Header dependencies: recompile the affected objects when a header changes.
$(OBJ)/demos.o $(OBJ)/example2.o $(OBJ)/SLIPLU.o $(OBJ)/qsx_basis.o $(OBJ)/update_demo.o $(OBJ)/perm.o $(OBJ)/ordering_ab.o: demos.h
$(OBJ)/slip_lu_update.o $(OBJ)/update_demo.o: slip_lu_update.h

# Link the demo programs from their objects.  'lib' is an order-only
# prerequisite (after the '|') so the library is kept up to date without
# forcing the programs to relink on every invocation.
$(BIN)/example: $(OBJ)/example.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/example.o $(CS)

$(BIN)/example2: $(OBJ)/example2.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/example2.o $(OBJ)/demos.o $(CS)

$(BIN)/SLIPLU: $(OBJ)/SLIPLU.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/SLIPLU.o $(OBJ)/demos.o $(CS)

$(BIN)/qsx_basis: $(OBJ)/qsx_basis.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/qsx_basis.o $(OBJ)/demos.o $(CS)

$(BIN)/update_demo: $(OBJ)/update_demo.o $(OBJ)/slip_lu_update.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/update_demo.o $(OBJ)/slip_lu_update.o $(OBJ)/demos.o $(CS)

$(BIN)/perm: $(OBJ)/perm.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/perm.o $(OBJ)/demos.o $(CS)

$(BIN)/ordering_ab: $(OBJ)/ordering_ab.o $(OBJ)/demos.o | lib $(BIN)
	$(CC) $(LDFLAGS) $(CF) -o $@ $(OBJ)/ordering_ab.o $(OBJ)/demos.o $(CS)

clean:
	- $(RM) -r $(OBJ) *.o

purge: distclean

distclean: clean
	- $(RM) -r $(BIN) *.a *.dSYM *.obj *.dll

.PHONY: default all lib clean purge distclean
