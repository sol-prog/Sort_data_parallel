import subprocess

def run_tests(prefix, nr_tests, threads):
	for i in range(nr_tests):
	    if prefix == "nvcc":
	        fname = prefix + "_" + "data_" + str(i) + ".txt"
	        program = "./" + prefix + "_test"
	        status = subprocess.call(program + " > " + fname, shell = True)
	    else:
	        for th in threads:
	            fname = prefix + "_" + "data_" + str(th) + "_" + str(i) + ".txt"
	            program = "./" + prefix + "_test"
	            status = subprocess.call(program + " " + str(th) + " > " + fname, shell = True)
		print("process " + str(i) + " finished")


nr_tests = 100
##compilers = ["clang", "gcc_463", "gcc_472", "nvcc"]
#compilers = ["nvcc"]
#compilers = ["clang"]
#compilers = ["gcc_463"]
compilers = ["gcc_472"]
threads = [1, 8]

for prefix in compilers:
	run_tests(prefix, nr_tests, threads)

print("Finish")

