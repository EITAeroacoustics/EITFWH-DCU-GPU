CC = mpicxx
CXXFLAGS = -std=c++17 -O2
CUDA_PATH = $(CUDA_HOME)
LDFLAGS = -lm -lfftw3 -L$(CUDA_PATH)/lib64 -lcudart
CUDA_INCLUDE = -I$(CUDA_PATH)/include
NVCC = $(CUDA_PATH)/bin/nvcc
NVCCFLAGS = -arch=gfx906 --gpu-max-threads-per-block=512

SRC_CPP = main.cpp Config.cpp Farrasat1A.cpp
SRC_CU = calTimeSignal.cu calFreqSignal.cu calFreqSignal2D.cu
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(SRC_CU:.cu=.o)


all: fwh_gpu

fwh_gpu: $(OBJ_CPP) $(OBJ_CU)
	$(CC) $(CXXFLAGS) $(CUDA_INCLUDE) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(CUDA_INCLUDE) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c -o $@ $<

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) fwh_gpu
