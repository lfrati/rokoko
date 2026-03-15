MAKEFLAGS += -j$(shell nproc)
CUDA_HOME ?= /usr/local/cuda-13.1
CUTLASS   ?= third_party/cutlass/include

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -I$(CUDA_HOME)/include -Isrc
NVFLAGS  = -std=c++17 -O3 -arch=native -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lpthread

.PHONY: clean
.DEFAULT_GOAL := rokoko

src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/cutlass_conv.o: src/cutlass_conv.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/main.o: src/main.cu src/g2p.h src/normalize.h src/weights.h src/kernels.h \
            src/bundle.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: src/main.o src/tts.cpp src/weights.cpp src/weights.h src/kernels.o src/cutlass_conv.o
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/tts.cpp src/weights.cpp \
		src/kernels.o src/cutlass_conv.o $(LDFLAGS) -o $@

clean:
	rm -f rokoko src/kernels.o src/main.o src/cutlass_conv.o
