# Average the data sets and calculate the standard deviation for each set
# In order to run the script you will need Python 2.7.3, Numpy and Matplotlib

from pylab import *

def get_data(prefix, nr_tests):
	tmp = []

	for i in range(nr_tests):
		fname = "data/" + prefix + "_" + str(i) + ".txt"
		tmp.append(np.loadtxt(fname))

	data = np.mean(tmp, 0)
	std = np.std(tmp, 0)
	return(data, std)

def gen_figure(prefix_avg_serial, prefix_std_serial, prefix_avg_tbb, prefix_std_tbb, prefix_avg_ms, prefix_std_ms, fig_title, fig_name):
	figure(1)
	plot(prefix_avg_serial[:,0], prefix_avg_serial[:,1], color="blue", linewidth=2.5, label = "std::sort")
	plot(prefix_avg_tbb[:,0], prefix_avg_tbb[:,1], color="red", linewidth=2.5, label = "tbb::parallel_sort")
	plot(prefix_avg_ms[:,0], prefix_avg_ms[:,1], color="green", linewidth=2.5, label = "parallel merge-sort")
	
	xlabel('Number of elements')
	ylabel('Time [ms]')
	title(fig_title)
	legend(loc='upper left')
	grid()

	errorbar(prefix_avg_serial[:,0], prefix_avg_serial[:,1], prefix_std_serial[:,1], color="blue")
	errorbar(prefix_avg_tbb[:,0], prefix_avg_tbb[:,1], prefix_std_tbb[:,1], color="red")
	errorbar(prefix_avg_ms[:,0], prefix_avg_ms[:,1], prefix_std_ms[:,1], color="green")
	
	savefig(fig_name, dpi=72)
	
	close()

def gen_figure_norm(prefix_avg_serial, prefix_std_serial, prefix_avg_tbb, prefix_std_tbb, prefix_avg_ms, prefix_std_ms, fig_title, fig_name):
	figure(1)
	#avoid division by "zero" by removing the first 50 elements
	cut = 50
	prefix_avg_serial = prefix_avg_serial[cut:,:]
	prefix_avg_tbb = prefix_avg_tbb[cut:,:]
	prefix_avg_ms = prefix_avg_ms[cut:,:]
	
	plot(prefix_avg_serial[:,0], prefix_avg_serial[:,1]/prefix_avg_tbb[:,1], color="blue", linewidth=2.5, label = "std::sort norm")
	plot(prefix_avg_ms[:,0], prefix_avg_ms[:,1]/prefix_avg_tbb[:,1], color="red", linewidth=2.5, label = "parallel merge-sort norm")
	
	xlabel('Number of elements')
	ylabel('Speedup')
	title(fig_title)
	legend(loc='upper left')
	grid()

	savefig(fig_name, dpi=72)
	
	close()

def gpu_gen_figure(prefix_avg_serial, prefix_std_serial, prefix_avg_tbb, prefix_std_tbb, prefix_avg_gpu, prefix_std_gpu, fig_title, fig_name):
	figure(1)
	plot(prefix_avg_serial[:,0], prefix_avg_serial[:,1], color="blue", linewidth=2.5, label = "std::sort")
	plot(prefix_avg_tbb[:,0], prefix_avg_tbb[:,1], color="red", linewidth=2.5, label = "tbb::parallel_sort")
	plot(prefix_avg_gpu[:,0], prefix_avg_gpu[:,1], color="green", linewidth=2.5, label = "GPU")
	
	xlabel('Number of elements')
	ylabel('Time [ms]')
	title(fig_title)
	legend(loc='upper left')
	grid()

	errorbar(prefix_avg_serial[:,0], prefix_avg_serial[:,1], prefix_std_serial[:,1], color="blue")
	errorbar(prefix_avg_tbb[:,0], prefix_avg_tbb[:,1], prefix_std_tbb[:,1], color="red")
	errorbar(prefix_avg_gpu[:,0], prefix_avg_gpu[:,1], prefix_std_gpu[:,1], color="green")
	
	savefig(fig_name, dpi=72)
	
	close()

def gpu_gen_figure_norm(prefix_avg_serial, prefix_std_serial, prefix_avg_tbb, prefix_std_tbb, prefix_avg_gpu, prefix_std_gpu, fig_title, fig_name):
	figure(1)	
	plot(prefix_avg_serial[:,0], prefix_avg_serial[:,1]/prefix_avg_gpu[:,1], color="blue", linewidth=2.5, label = "std::sort norm")
	plot(prefix_avg_tbb[:,0], prefix_avg_tbb[:,1]/prefix_avg_gpu[:,1], color="red", linewidth=2.5, label = "tbb::parallel_sort norm")
	
	xlabel('Number of elements')
	ylabel('Speedup')
	title(fig_title)
	legend(loc='upper left')
	grid()

	savefig(fig_name, dpi=72)
	
	close()

nr_tests = 100

clang_avg_tbb, clang_std_tbb = get_data("clang_data", nr_tests)
gcc_463_avg_tbb, gcc_463_std_tbb = get_data("gcc_463_data", nr_tests)
gcc_472_avg_tbb, gcc_472_std_tbb = get_data("gcc_472_data", nr_tests)

nvcc_avg, nvcc_std = get_data("nvcc_data", nr_tests)

clang_avg_serial, clang_std_serial = get_data("clang_data_serial", nr_tests)
gcc_463_avg_serial, gcc_463_std_serial = get_data("gcc_463_data_serial", nr_tests)
gcc_472_avg_serial, gcc_472_std_serial = get_data("gcc_472_data_serial", nr_tests)

clang_avg_ms, clang_std_ms = get_data("clang_data_ms", nr_tests)
gcc_463_avg_ms, gcc_463_std_ms = get_data("gcc_463_data_ms", nr_tests)
gcc_472_avg_ms, gcc_472_std_ms = get_data("gcc_472_data_ms", nr_tests)

# TBB parallel_sort vs my parallel merge-sort
gen_figure(clang_avg_serial, clang_std_serial, clang_avg_tbb, clang_std_tbb, clang_avg_ms, clang_std_ms, "Clang-3.3svn", "clang_cpu_sort.png")
gen_figure(gcc_472_avg_serial, gcc_472_std_serial, gcc_472_avg_tbb, gcc_472_std_tbb, gcc_472_avg_ms, gcc_472_std_ms, "GCC-4.7.2", "gcc_472_cpu_sort.png")
gen_figure(gcc_463_avg_serial, gcc_463_std_serial, gcc_463_avg_tbb, gcc_463_std_tbb, gcc_463_avg_ms, gcc_463_std_ms, "GCC-4.6.3", "gcc_463_cpu_sort.png")

gen_figure_norm(clang_avg_serial, clang_std_serial, clang_avg_tbb, clang_std_tbb, clang_avg_ms, clang_std_ms, "Clang-3.3svn", "clang_cpu_norm.png")
gen_figure_norm(gcc_472_avg_serial, gcc_472_std_serial, gcc_472_avg_tbb, gcc_472_std_tbb, gcc_472_avg_ms, gcc_472_std_ms, "GCC-4.7.2", "gcc_472_cpu_norm.png")
gen_figure_norm(gcc_463_avg_serial, gcc_463_std_serial, gcc_463_avg_tbb, gcc_463_std_tbb, gcc_463_avg_ms, gcc_463_std_ms, "GCC-4.6.3", "gcc_463_cpu_norm.png")

# TBB parallel_sort vs CUDA
gpu_gen_figure(clang_avg_serial, clang_std_serial, clang_avg_tbb, clang_std_tbb, nvcc_avg, nvcc_std, "Clang-3.3svn CPU vs GPU", "clang_cpu_gpu_sort.png")
gpu_gen_figure(gcc_472_avg_serial, gcc_472_std_serial, gcc_472_avg_tbb, gcc_472_std_tbb, nvcc_avg, nvcc_std, "GCC-4.7.2 CPU vs GPU", "gcc_472_cpu_gpu_sort.png")
gpu_gen_figure(gcc_463_avg_serial, gcc_463_std_serial, gcc_463_avg_tbb, gcc_463_std_tbb, nvcc_avg, nvcc_std, "GCC-4.6.3 CPU vs GPU", "gcc_463_cpu_gpu_sort.png")

gpu_gen_figure_norm(clang_avg_serial, clang_std_serial, clang_avg_tbb, clang_std_tbb, nvcc_avg, nvcc_std, "Clang-3.3svn CPU vs GPU", "clang_cpu_gpu_norm.png")
gpu_gen_figure_norm(gcc_472_avg_serial, gcc_472_std_serial, gcc_472_avg_tbb, gcc_472_std_tbb, nvcc_avg, nvcc_std, "GCC-4.7.2 CPU vs GPU", "gcc_472_cpu_gpu_norm.png")
gpu_gen_figure_norm(gcc_463_avg_serial, gcc_463_std_serial, gcc_463_avg_tbb, gcc_463_std_tbb, nvcc_avg, nvcc_std, "GCC-4.6.3 CPU vs GPU", "gcc_463_cpu_gpu_norm.png")



