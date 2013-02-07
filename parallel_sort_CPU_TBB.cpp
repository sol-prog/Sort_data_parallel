#include <iostream>
#include <vector>
#include <random>
#include <boost/chrono.hpp>
#include <algorithm>
#include <ctime>
#include "utils.h"
#include <tbb/parallel_sort.h>

// Sort data on a single CPU (std::sort)
double run_tests_std_sort(std::vector<double> &V) {

    auto start = boost::chrono::steady_clock::now();
    
    std::sort(std::begin(V), std::end(V));
    
    auto end = boost::chrono::steady_clock::now();
    
    return boost::chrono::duration <double, boost::milli> (end - start).count();
}

// Use Intel's TBB to sort the data on all available CPUs
double run_tests_tbb(std::vector<double> &V) {
    auto start = boost::chrono::steady_clock::now();
    
    tbb::parallel_sort(std::begin(V), std::end(V));
    
    auto end = boost::chrono::steady_clock::now();
    
    return boost::chrono::duration <double, boost::milli> (end - start).count();
}


int main(int argc, char **argv) {
    std::vector<double> V;
    
    //use the system time to create a random seed
    unsigned int seed = (unsigned int) time(nullptr);
    
    size_t parts = 0;
    
    if(argc > 2) {
        std::cout << "ERROR! Correct program usage:" << std::endl;
        std::cout << argv[0] << " nr_parts" << std::endl;
        std::exit(1);
    }
    
    // Get the number of parts
    if(argc == 2) {
        std::string s(argv[1]);
        parts = (size_t) std::stoi(s);    
    }
    
    size_t step = 10;
    size_t mem = 10000000;

    for(size_t i = 16; i <= mem; i = 2 * step, step *= 1.1) {
        //Fill V with random numbers in the range [0,1]:
        V.resize(i);
        rnd_fill(V, 0.0, 1.0, seed);
        if(parts == 1) {
            // serial sort
            std::cout << i << "\t" << run_tests_std_sort(V) << std::endl;
        }
        else {
            // parallel sort (Intel TBB)
            std::cout << i << "\t" << run_tests_tbb(V) << std::endl;        
        }
    }
    
    return 0;
}

