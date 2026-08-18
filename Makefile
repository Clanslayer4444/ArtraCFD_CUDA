#***************************************************************************#
#                          ArtraCFD Makefile                                #
#                          (Multi-Architecture Build)                       #
#***************************************************************************#

SHELL := /bin/bash
INSTALL := install
INSTALLDATA := $(INSTALL) -m 644
prefix = ~
bindir = $(prefix)/Bin
BINNAME := artracfd
srcdir = .

#============================================================================#
# Build Configuration                                                        #
#   BUILD=cpu    - CPU-only, no CUDA dependency                              #
#   BUILD=gpu    - Single-GPU accelerated (C with CUDA kernels)              #
#   BUILD=hpc    - MPI + GPU (Multi-CPU + 1 GPU)                            #
#============================================================================#
BUILD ?= gpu

#============================================================================#
# Compilers                                                                  #
#============================================================================#
CC := gcc
CXX := g++
NVCC := nvcc
MPICC ?= mpicc

#------------------ Base Flags ------------------#
CFLAGS    := -Wall -Wextra -O2 -std=c99 -pedantic
CXXFLAGS  := -Wall -Wextra -O2 -std=c++11
NVCCFLAGS := -O2 -std=c++11 -rdc=true
INCLUDES  :=
LFLAGS    :=
LIBS      := -lm

#------------------ Architecture Flags ------------------#
# CPU only
CFLAGS_cpu    := $(CFLAGS) -fopenmp
NVCCFLAGS_cpu :=
LFLAGS_cpu    :=
LIBS_cpu      := -lm -lgomp

# GPU accelerated
CFLAGS_gpu    := $(CFLAGS) -DCUDA_ENABLED -fopenmp
NVCCFLAGS_gpu := $(NVCCFLAGS) -DCUDA_ENABLED
LFLAGS_gpu    := -L/usr/local/cuda/lib64
LIBS_gpu      := -lm -lcudart -lgomp

# HPC: MPI + GPU
CFLAGS_hpc    := $(CFLAGS) -DCUDA_ENABLED -DMPI_ENABLED -fopenmp
NVCCFLAGS_hpc := $(NVCCFLAGS) -DCUDA_ENABLED -DMPI_ENABLED
LFLAGS_hpc    := -L/usr/local/cuda/lib64
# MPI C++ linker needed for CUDA + MPI combined builds
MPICXX       ?= mpicxx
LIBS_hpc      := -lm -lcudart -lgomp

#------------------ Select active flags ------------------#
CFLAGS_ARCH    := $(CFLAGS_$(BUILD))
NVCCFLAGS_ARCH := $(NVCCFLAGS_$(BUILD))
LFLAGS_ARCH    := $(LFLAGS_$(BUILD))
LIBS_ARCH      := $(LIBS_$(BUILD))

#------------------ Source Files ------------------#
SRCS_C   := $(wildcard *.c)
SRCS_CU  := program_entrance.cu linear_system_gpu.cu gpu_fluid_dynamics.cu \
            gpu_state.cu gpu_convective_flux.cu gpu_eigen.cu \
            gpu_diffusive_flux.cu gpu_ibm.cu

# MPI interface included only for hpc build
ifeq ($(BUILD),hpc)
    SRCS_C += mpi_interface.c
endif

SRCS     := $(SRCS_C) $(SRCS_CU)
OBJS_C   := $(SRCS_C:.c=.o)
OBJS_CU  := $(SRCS_CU:.cu=.o)
OBJS     := $(OBJS_C) $(OBJS_CU)
CLEANLIST := $(OBJS) $(BINNAME)

#------------------ Linker ------------------#
# For hpc, use MPI C++ linker to handle CUDA + MPI symbol resolution
ifeq ($(BUILD),hpc)
    LINKER    := $(MPICXX)
    LINKFLAGS := $(NVCCFLAGS_ARCH) $(LFLAGS_ARCH)
    LINKLIBS  := $(LIBS_ARCH)
else ifeq ($(BUILD),gpu)
    LINKER    := $(NVCC)
    LINKFLAGS := $(NVCCFLAGS_ARCH) $(LFLAGS_ARCH)
    LINKLIBS  := $(LIBS_ARCH)
else
    LINKER    := $(CC)
    LINKFLAGS := $(CFLAGS_ARCH) $(INCLUDES) $(LFLAGS_ARCH)
    LINKLIBS  := $(LIBS_ARCH)
endif

#============================================================================#
# Build Rules                                                                #
#============================================================================#
.PHONY: all install uninstall clean test_verify

all: $(BINNAME)
	@echo "  ArtraCFD built: BUILD=$(BUILD)"

$(BINNAME): $(OBJS)
	$(LINKER) $(LINKFLAGS) -o $@ $(OBJS) $(LINKLIBS)

%.o: %.c
	$(CC) $(CFLAGS_ARCH) $(INCLUDES) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS_ARCH) -c -o $@ $<

install: $(BINNAME)
	@mkdir -p $(bindir)
	$(INSTALL) $(BINNAME) $(bindir)/$(BINNAME)

uninstall:
	$(RM) $(bindir)/$(BINNAME)

clean:
	@echo "  cleaning..."
	@- $(RM) $(CLEANLIST)

#============================================================================#
# Verification Test                                                          #
#============================================================================#
test_verify: $(BINNAME)
	@echo "  Running CPU-vs-GPU verification..."
	cd test && bash run_verify.sh

#============================================================================#
# Dependency Generation (C files)                                            #
#============================================================================#
DPND := $(SRCS_C:.c=.d)

$(DPND): %.d: %.c
	@set -e; rm -f $@; \
		$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

ifneq ($(MAKECMDGOALS),clean)
    -include $(DPND)
endif

CLEANLIST += $(DPND)
