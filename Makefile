CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS = -lm

all: symnmf

symnmf: symnmf.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o $(LIBS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf