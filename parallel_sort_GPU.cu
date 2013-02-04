#include <iostream>
#include <vector>
#include <ctime>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "utils.h"


int main() {
    std::vector<double> V;
    thrust::device_vector<double> d_V;
    
    //use the system time to create a random seed
    unsigned int seed = (unsigned int) time(NULL);
    
    size_t step = 10;
    size_t mem = 10000000;

    for(size_t i = 16; i <= mem; i = 2 * step, step *= 1.1) {
        //Fill V with random numbers in the range [0,1]:
        V.resize(i);
        rnd_fill(V, 0.0, 1.0, seed);
        d_V = V;

	    cudaEvent_t start, stop;
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);

	    //Start recording
	    cudaEventRecord(start,0);
        
        thrust::stable_sort(d_V.begin(), d_V.end());
        
	    //Stop recording
	    cudaEventRecord(stop,0);
	    cudaEventSynchronize(stop);
	    float elapsedTime;
	    cudaEventElapsedTime(&elapsedTime, start, stop);

	    cudaEventDestroy(start);
	    cudaEventDestroy(stop);

	    std::cout << i << "\t" << elapsedTime << std::endl;
    }
    
    return 0;
}

