#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "oclhelpers.h"

int main(void)
{
	struct ocl_environ environ;
	unsigned int count = 4096;
	size_t local, global;
	cl_kernel kernel = NULL;
	cl_int err;
	
	int retval = 0;
	float dataa[count];
	float datab[count];
	float result[count];
	int i, correct;

	cl_mem d_result = NULL;
	cl_mem d_dataa = NULL;
	cl_mem d_datab = NULL;

	ocl_kernel_arg args[5] = {
		{&d_result, sizeof(cl_mem)},
		{&d_dataa, sizeof(cl_mem)},
		{&d_datab, sizeof(cl_mem)},
		{&count, sizeof(unsigned int)},
		{NULL, 0}};

	srandom(time(0));

	for (i = 0; i < count; i++) {
		dataa[i] = rand();
		datab[i] = rand();
	}

	err = ocl_default_environ(&environ);
	ocl_check_error(&environ, err, "setup environ");
	
	ocl_compile_file(&environ, "vectormult.cl");
	ocl_check_error(&environ, err, "compile program");


	kernel = clCreateKernel(environ.program, "vector_mult", &err);
	ocl_check_error(&environ, err, "create kernel");

	err = ocl_alloc_output(&environ, &d_result, sizeof(float) * count);
	ocl_check_error(&environ, err, "alloc result");
	
	err = ocl_copy_input(&environ, &d_dataa, dataa, sizeof(float) * count);
	ocl_check_error(&environ, err, "copy dataa");
	
	err = ocl_copy_input(&environ, &d_datab, datab, sizeof(float) * count);
	ocl_check_error(&environ, err, "copy datab");
	
	err = ocl_set_kernel_args(kernel, args);
	ocl_check_error(&environ, err, "set kernel args");
	
	local = 1024;
	global = count;
	err = clEnqueueNDRangeKernel(environ.commands, kernel, 1, NULL,
				&global, &local, 0, NULL, NULL);
	ocl_check_error(&environ, err, "enqueue kernel");
	clFinish(environ.commands);

	err = clEnqueueReadBuffer(environ.commands, d_result, CL_TRUE, 0,
				sizeof(float) * count, result, 0, NULL, NULL);
	ocl_check_error(&environ, err, "copy result");
	
	correct = 0;
	for (i = 0; i < count; i++) {
		if (result[i] == dataa[i] * datab[i])
			correct++;
	}
	printf("%d correct, %d incorrect\n", correct, count - correct);

	clReleaseMemObject(d_result);
	clReleaseMemObject(d_dataa);
	clReleaseMemObject(d_datab);
	clReleaseKernel(kernel);
	ocl_destroy_environ(&environ);

	return retval;
}
