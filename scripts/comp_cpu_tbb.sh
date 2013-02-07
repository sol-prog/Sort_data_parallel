# We assume that Intel's TBB is installed in /usr/tbb/...
clang++ -O3 -std=c++11 -stdlib=libc++ -I/usr/tbb41/include -L/usr/tbb41/lib/intel64/cc4.1.0_libc2.4_kernel2.6.16.21 utils.cpp parallel_sort_CPU_TBB.cpp -lboost_chrono -lboost_system -ltbb -o clang_test_tbb
g++ -O3 -std=c++0x -I/usr/tbb41/include -L/usr/tbb41/lib/intel64/cc4.1.0_libc2.4_kernel2.6.16.21 utils.cpp parallel_sort_CPU_TBB.cpp -lboost_chrono -lboost_system -ltbb -o gcc_463_test_tbb
g++-4.7.2 -O3 -std=c++11 -I/usr/tbb41/include -L/usr/tbb41/lib/intel64/cc4.1.0_libc2.4_kernel2.6.16.21 utils.cpp parallel_sort_CPU_TBB.cpp -lboost_chrono -lboost_system -ltbb -o gcc_472_test_tbb
