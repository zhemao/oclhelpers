CFLAGS=-Wall -O2
LDFLAGS=-lOpenCL

ocltest: ocltest.c oclHelpers.o
	$(CC) $(CFLAGS) ocltest.c oclHelpers.o $(LDFLAGS) -o ocltest

%.o: %.c %.h
