import subprocess

#Run parallel tests (Intel TBB)
def run_tests(prefix, nr_tests):
	for i in range(nr_tests):
		fname = prefix + "_data_" + str(i) + ".txt"
		program = "./" + prefix + "_test_tbb"		
		status = subprocess.call(program + " > " + fname, shell = True)

#Run parallel tests (my merge_sort)
def run_tests_cpu(prefix, nr_tests):
	for i in range(nr_tests):
		fname = prefix + "_data_ms_" + str(i) + ".txt"
		program = "./" + prefix + "_test_ms 8"		
		status = subprocess.call(program + " > " + fname, shell = True)
	
#Run serial tests
def run_tests_serial(prefix, nr_tests):
	for i in range(nr_tests):
		fname = prefix + "_data_serial_" + str(i) + ".txt"
		program = "./" + prefix + "_test_tbb 1"		
		status = subprocess.call(program + " > " + fname, shell = True)
	
nr_tests = 100
compilers = ["clang", "gcc_463", "gcc_472", "nvcc"]


for prefix in compilers:
	run_tests(prefix, nr_tests)
	print("Finished processing " + prefix + " tests")

for prefix in compilers[:len(compilers)-1]:
	run_tests_serial(prefix, nr_tests)
	print("Finished processing " + prefix + " tests")

for prefix in compilers[:len(compilers)-1]:
	run_tests_cpu(prefix, nr_tests)
	print("Finished processing " + prefix + " tests")

print("FINISH")
