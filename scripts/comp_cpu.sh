clang++ -O3 -std=c++11 -stdlib=libc++ utils.cpp parallel_sort_CPU.cpp -lboost_chrono -lboost_system -o clang_test
g++ -O3 -std=c++0x utils.cpp parallel_sort_CPU.cpp -lboost_chrono -lboost_system -o gcc_463_test
g++-4.7.2 -O3 -std=c++11 utils.cpp parallel_sort_CPU.cpp -lboost_chrono -lboost_system -o gcc_472_test

