#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <boost/chrono.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

//Fill a vector with random numbers in the range [lower, upper]
void rnd_fill(thrust::host_vector<double> &V, const double lower, const double upper, const unsigned int seed) {

    //Create a unique seed for the random number generator
    srand(time(NULL));
    
    size_t elem = V.size();
    for( size_t i = 0; i < elem; ++i){
        V[i] = (double) rand() / (double) RAND_MAX;
    }
}



int main() {
    thrust::host_vector<double> V;
    thrust::device_vector<double> d_V;
    
    //use the system time to create a random seed
    unsigned int seed = (unsigned int) time(NULL);
    
    size_t step = 10;
    size_t mem = 10000000;

    for(size_t i = 16; i <= mem; i = 2 * step, step *= 1.1) {
        //Fill V with random numbers in the range [0,1]:
        V.resize(i);
        rnd_fill(V, 0.0, 1.0, seed);
        
        boost::chrono::steady_clock::time_point start_cpu = boost::chrono::steady_clock::now();
        d_V = V; // Transfer data to the GPU
        boost::chrono::steady_clock::time_point end_cpu = boost::chrono::steady_clock::now();
        double dt1 = boost::chrono::duration <double, boost::milli> (end_cpu - start_cpu).count();

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
	    
        start_cpu = boost::chrono::steady_clock::now();
        V = d_V; // Transfer data to the CPU
        end_cpu = boost::chrono::steady_clock::now();
        double dt2 = boost::chrono::duration <double, boost::milli> (end_cpu - start_cpu).count();
	    

	    //std::cout << i << "\t" << elapsedTime << "\t" << dt1 + dt2 << std::endl;
	    std::cout << i << "\t" << elapsedTime + dt1 + dt2 << std::endl;
    }
    
    return 0;
}

