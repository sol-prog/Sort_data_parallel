//Split "mem" into "parts", e.g. if mem = 10 and parts = 4 you will have: 0,2,4,6,10
//if possible the function will split mem into equal chuncks, if not 
//the last chunck will be slightly larger
std::vector<size_t> bounds(size_t parts, size_t mem);

//Fill a vector with random numbers in the range [lower, upper]
void rnd_fill(std::vector<double> &V, const double lower, const double upper, const unsigned int seed);



