all: libaod3d.so

FFLAGS := -fimplicit-none

libaod3d.so: aod3d.h aod3d.c Makefile
	gcc -std=c11 -shared -o libaod3d.so aod3d.c -I./ -O2 -lm -fPIC

clean:
	rm -fv *.so

