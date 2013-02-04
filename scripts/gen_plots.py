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
	    
def gen_figure(cpu_data1, cpu_std1,cpu_data8, cpu_std8,nvcc_data, nvcc_std, fig_title,fig_name):
	figure(1)
	plot(cpu_data1[:,0],cpu_data1[:,1], color="blue", linewidth=2.5, label = "CPU 1 thread")
	plot(cpu_data8[:,0],cpu_data8[:,1], color="red", linewidth=2.5, label = "CPU 8 threads")
	plot(nvcc_data[:,0],nvcc_data[:,1], color="green", linewidth=2.5, label = "GPU")

	xlabel('Number of elements')
	ylabel('Time [ms]')
	title(fig_title)
	legend(loc='upper left')
	grid()

	errorbar(cpu_data1[:,0],cpu_data1[:,1], cpu_std1[:,1], color="blue")
	errorbar(cpu_data8[:,0],cpu_data8[:,1], cpu_std8[:,1], color="red")
	errorbar(nvcc_data[:,0],nvcc_data[:,1], nvcc_std[:,1], color="green")

	savefig(fig_name, dpi=72)

	close()

def gen_figure_norm(cpu_data1, cpu_data8, fig_title,fig_name):
	figure(1)
	plot(cpu_data1[:,0],cpu_data1[:,1], color="blue", linewidth=2.5, label = "CPU-1/GPU")
	plot(cpu_data8[:,0],cpu_data8[:,1], color="red", linewidth=2.5, label = "CPU-8/GPU")

	xlabel('Number of elements')
	ylabel('Speedup')
	title(fig_title)
	legend(loc='upper left')
	grid()

	savefig(fig_name, dpi=72)

	close()

def gen_figure_norm_cpu(cpu_data1, fig_title,fig_name):
	figure(1)
	plot(cpu_data1[:,0],cpu_data1[:,1], color="blue", linewidth=2.5, label = "CPU-1/CPU-8")

	xlabel('Number of elements')
	ylabel('Speedup')
	title(fig_title)
	legend(loc='upper left')
	grid()

	savefig(fig_name, dpi=72)

	close()

def gen_plot(clang_prefix1, gcc47_prefix1, gcc46_prefix1,clang_prefix8, gcc47_prefix8, gcc46_prefix8, nvcc_prefix,  nr_tests, descr):	
	clang_data1, clang_std1 = get_data(clang_prefix1, nr_tests)
	gcc47_data1, gcc47_std1 = get_data(gcc47_prefix1, nr_tests)
	gcc46_data1, gcc46_std1 = get_data(gcc46_prefix1, nr_tests)

	clang_data8, clang_std8 = get_data(clang_prefix8, nr_tests)
	gcc47_data8, gcc47_std8 = get_data(gcc47_prefix8, nr_tests)
	gcc46_data8, gcc46_std8 = get_data(gcc46_prefix8, nr_tests)
	
	nvcc_data, nvcc_std = get_data(nvcc_prefix, nr_tests)
	
	gen_figure(clang_data1, clang_std1,clang_data8, clang_std8,nvcc_data, nvcc_std, "Clang-3.3svn CPU vs GPU", "Clang_all.png")
	gen_figure(gcc47_data1, gcc47_std1,gcc47_data8, gcc47_std8,nvcc_data, nvcc_std, "Gcc-4.7.2 CPU vs GPU", "Gcc472_all.png")
	gen_figure(gcc46_data1, gcc46_std1,gcc46_data8, gcc46_std8,nvcc_data, nvcc_std, "Gcc-4.6.3 CPU vs GPU", "Gcc463_all.png")
	
	clang_norm1 = clang_data1
	clang_norm1[:,1] = clang_data1[:,1]/nvcc_data[:,1]

	clang_norm8 = clang_data8
	clang_norm8[:,1] = clang_data8[:,1]/nvcc_data[:,1]

	gcc47_norm1 = gcc47_data1
	gcc47_norm1[:,1] = gcc47_data1[:,1]/nvcc_data[:,1]

	gcc47_norm8 = gcc47_data8
	gcc47_norm8[:,1] = gcc47_data8[:,1]/nvcc_data[:,1]

	gcc46_norm1 = gcc46_data1
	gcc46_norm1[:,1] = gcc46_data1[:,1]/nvcc_data[:,1]

	gcc46_norm8 = gcc46_data8
	gcc46_norm8[:,1] = gcc46_data8[:,1]/nvcc_data[:,1]
	
	gen_figure_norm(clang_norm1, clang_norm8, "Clang-3.3svn CPU normalized with GPU","Clang_normalized.png")
	gen_figure_norm(gcc47_norm1, gcc47_norm8, "Gcc-4.7.2 CPU normalized with GPU","Gcc472_normalized.png")
	gen_figure_norm(gcc46_norm1, gcc46_norm8, "Gcc-4.6.3 CPU normalized with GPU","Gcc463_normalized.png")
	
	clang_norm1cpu = clang_data1
	clang_norm1cpu[:,1] = clang_data1[:,1]/clang_data8[:,1]
	
	gcc47_norm1cpu = gcc47_data1
	gcc47_norm1cpu[:,1] = gcc47_data1[:,1]/gcc47_data8[:,1]
	
	gcc46_norm1cpu = gcc46_data1
	gcc46_norm1cpu[:,1] = gcc46_data1[:,1]/gcc46_data8[:,1]
	
	gen_figure_norm_cpu(clang_norm1cpu, "Clang-3.3svn CPU normalized","Clang_normalized_cpu.png")
	gen_figure_norm_cpu(gcc47_norm1cpu, "Gcc-4.7.2 CPU normalized","Gcc472_normalized_cpu.png")
	gen_figure_norm_cpu(gcc46_norm1cpu, "Gcc-4.6.3 CPU normalized","Gcc463_normalized_cpu.png")
	
	

nr_tests = 100
gen_plot("clang_data_1", "gcc_472_data_1", "gcc_463_data_1","clang_data_8", "gcc_472_data_8", "gcc_463_data_8", "nvcc_data", nr_tests, "Clang 3.2svn")

