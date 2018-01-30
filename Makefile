CC = mpicc
CXX = mpicxx
LDFLAGS = -lpng -lm -fopenmp
CFLAGS = -O3 -march=native -std=gnu99
CXXFLAGS = -O3 -march=native -std=gnu++11
TARGETS = ms_mpi_static ms_mpi_dynamic ms_omp ms_hybrid 

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
