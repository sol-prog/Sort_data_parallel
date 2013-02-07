nvcc -O3 -arch=sm_21 parallel_sort_GPU.cu -lboost_chrono -lboost_system -o nvcc_test_tbb

