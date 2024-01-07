CUDA_PATH     ?= /usr/local/cuda
CXX            = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
NVPROF         = $(CUDA_PATH)/bin/nvprof

DEBUG = false

ifeq ($(DEBUG), true)
	CXXDEBUGFLAG += -g
	NVDEBUGFLAG += -g -G -dopt=on
else
	CXXDEBUGFLAG += -O3
	NVDEBUGFLAG += -O3
endif

CXXFLAGS = -Wall $(CXXDEBUGFLAG) -MMD -MP

# GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS  = 
NVCCFLAGS      = $(NVDEBUGFLAG) $(GENCODE_FLAGS) -m64 -MMD -MP -rdc=true

# Target binary
TARGET = main

# Source files
SRC_CXX = $(wildcard *.cc)
SRC_CU = $(wildcard *.cu)

# Object files
OBJ_CXX = $(SRC_CXX:.cc=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

# Dependency files
DEP_CXX = $(OBJ_CXX:.o=.d)
DEP_CU = $(OBJ_CU:.o=.d)

# All object files
OBJ = $(OBJ_CXX) $(OBJ_CU)

# Default target
all: $(TARGET)

# Linking the target with object files
main: $(filter-out test.o, $(OBJ))
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compiling .cc to .o
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

# Compiling .cu to .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJ) $(DEP_CXX) $(DEP_CU)

# Include the dependency files
-include $(DEP_CXX)
-include $(DEP_CU)

.PHONY: all clean
