g++ -c -O3 -std=c++0x utils.cpp
nvcc -O3 -arch=sm_21 utils.o parallel_sort_GPU.cu -o nvcc_test

