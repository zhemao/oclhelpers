CFLAGS=-Wall -O2
LDFLAGS=-lOpenCL

ocltest: ocltest.c oclhelpers.o
	$(CC) $(CFLAGS) ocltest.c oclhelpers.o $(LDFLAGS) -o ocltest

%.o: %.c %.h
