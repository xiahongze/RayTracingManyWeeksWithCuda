CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVPROF         = $(CUDA_PATH)/bin/nvprof

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -O3
# GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS  = 

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h interval.h image_utils.h

main: main.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main main.o cmd_parser.o

cmd_parser.o: cmd_parser.h cmd_parser.cpp
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cmd_parser.o -c cmd_parser.cpp

main.o: $(SRCS) $(INCS) cmd_parser.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

out.jpg: main
	rm -f out.jpg
	./main

profile_basic: main
	$(NVPROF) ./main

# use nvprof --query-metrics
profile_metrics: main
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./main

clean:
	rm -f main main.o out.jpg

PHONY: clean

default: main
