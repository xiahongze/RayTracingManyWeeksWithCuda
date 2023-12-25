CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -O3
# GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS  = 

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h interval.h

main: main.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main main.o

main.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

out.ppm: main
	rm -f out.ppm
	./main > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	convert out.ppm out.jpg

profile_basic: main
	nvprof ./main > out.ppm

# use nvprof --query-metrics
profile_metrics: main
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./main > out.ppm

clean:
	rm -f main main.o out.ppm out.jpg

PHONY: clean
