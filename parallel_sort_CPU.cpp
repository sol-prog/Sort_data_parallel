#include <iostream>
#include <vector>
#include <random>
#include <boost/chrono.hpp>
#include <algorithm>
#include <thread>
#include <ctime>
#include "utils.h"

//Use std::sort
void test_sort(std::vector<double> &V, size_t left, size_t right) {    
    std::sort(std::begin(V) + left, std::begin(V) + right);
}

//Merge V[n0:n1] with V[n2:n3]. The result is put back to V[n0:n3]
void par_merge(std::vector<double> &V, size_t n0, size_t n1, size_t n2, size_t n3) {
    std::vector<double> aux(n1 - n0 + n3 - n2);
    std::merge(std::begin(V) + n0, std::begin(V) + n1, std::begin(V) + n2, std::begin(V) + n3, std::begin(aux));
    std::copy(std::begin(aux), std::end(aux), std::begin(V) + n0);
}

double run_tests(std::vector<double> &V, size_t parts, size_t mem) {

    //Split the data in "parts" pieces and sort each piece in a separate thread
    std::vector<size_t> bnd = bounds(parts, mem);
    std::vector<std::thread> thr;
    
    auto start = boost::chrono::steady_clock::now();    
    
    //Launch "parts" threads
    for(size_t i = 0; i < parts; ++i) {
        thr.push_back(std::thread(test_sort, std::ref(V), bnd[i], bnd[i + 1]));
    }
    
    for(auto &t : thr) {
        t.join();
    }
    
    //Merge data
    while(parts >= 2) {
        std::vector<size_t> limits;
        std::vector<std::thread> th;
        for(size_t i = 0; i < parts - 1; i += 2) {
            th.push_back(std::thread(par_merge, std::ref(V), bnd[i], bnd[i + 1], bnd[i + 1], bnd[i + 2]));
            
            size_t naux = limits.size();
            if(naux > 0) {
                if(limits[naux - 1] != bnd[i]) {
                    limits.push_back(bnd[i]);
                }
                limits.push_back(bnd[i + 2]);
            }
            else {
                limits.push_back(bnd[i]);
                limits.push_back(bnd[i + 2]);
            } 
        }
        
        for(auto &t : th) {
            t.join();
        }
                
        parts /= 2;
        bnd = limits;
    }
    auto end = boost::chrono::steady_clock::now();
    
    return boost::chrono::duration <double, boost::milli> (end - start).count();
}


int main(int argc, char **argv) {
    std::vector<double> V;
    
    //use the system time to create a random seed
    unsigned int seed = (unsigned int) time(nullptr);
    
    size_t parts;
    
    if(argc != 2) {
        std::cout << "ERROR! Correct program usage:" << std::endl;
        std::cout << argv[0] << " nr_parts" << std::endl;
        std::exit(1);
    }
    
    // Get the number of parts
    std::string s(argv[1]);
    //try {
        parts = (size_t) std::stoi(s);
    //}
    //catch (std::exception &ex) {
    //    std::cout << "ERROR! Please use a valid numerical input!" << std::endl;
    //    std::exit(1);
    //}
    
    size_t step = 10;
    size_t mem = 10000000;

    for(size_t i = 16; i <= mem; i = 2 * step, step *= 1.1) {
        //Fill V with random numbers in the range [0,1]:
        V.resize(i);
        rnd_fill(V, 0.0, 1.0, seed);
        std::cout << i << "\t" << run_tests(V, parts, i) << std::endl;        
    }
    
    return 0;
}

