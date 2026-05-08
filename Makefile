# Makefile -- Build the symnmf standalone C executable.
#
# Usage:
#   make          build ./symnmf
#   make clean    remove compiled objects and the binary

CC     = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS   = -lm

all: symnmf

# Link the final executable from the compiled object file.
symnmf: symnmf.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o $(LIBS)

# Compile symnmf.c without defining SYMNMF_PYTHON_MODULE so that main() is included.
symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf
