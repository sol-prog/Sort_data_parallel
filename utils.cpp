#include <vector>
#include <random>

//Split "mem" into "parts", e.g. if mem = 10 and parts = 4 you will have: 0,2,4,6,10
//if possible the function will split mem into equal chuncks, if not 
//the last chunck will be slightly larger
std::vector<size_t> bounds(size_t parts, size_t mem) {
    std::vector<size_t>bnd(parts + 1);
    size_t delta = mem / parts;
    size_t reminder = mem % parts;
    size_t N1 = 0, N2 = 0;
    bnd[0] = N1;
    for (size_t i = 0; i < parts; ++i) {
        N2 = N1 + delta;
        if (i == parts - 1)
            N2 += reminder;
        bnd[i + 1] = N2;
        N1 = N2;
    }
    return bnd;
}


//Fill a vector with random numbers in the range [lower, upper]
void rnd_fill(std::vector<double> &V, const double lower, const double upper, const unsigned int seed) {

    //use the default random engine and an uniform distribution
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> distr(lower, upper);

    for( auto &elem : V){
        elem = distr(eng);
    }
}

