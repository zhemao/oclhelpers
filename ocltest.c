#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "oclHelpers.h"

int main(void)
{
	struct ocl_environ environ;
	unsigned int count = 4096;
	size_t local, global;
	cl_program program = NULL;
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
	if (err != CL_SUCCESS) {
		ocl_perror("default_environ", err);
		retval = -1;
		goto cleanup;
	}
	
	ocl_compile_file("vectormult.cl", &environ, &program);
	if (err != CL_SUCCESS) {
		ocl_perror("compile_file", err);
		retval = -1;
		goto cleanup;
	}


	kernel = clCreateKernel(program, "vector_mult", &err);
	if (err != CL_SUCCESS) {
		ocl_perror("compile_file", err);
		retval = -1;
		goto cleanup;
	}

	err = ocl_alloc_output(&environ, &d_result, sizeof(float) * count);
	if (err != CL_SUCCESS) {
		ocl_perror("create result buffer", err);
		retval = -1;
		goto cleanup;
	}
	err = ocl_copy_input(&environ, &d_dataa, dataa, sizeof(float) * count);
	if (err != CL_SUCCESS) {
		ocl_perror("copy dataa", err);
		retval = -1;
		goto cleanup;
	}
	err = ocl_copy_input(&environ, &d_datab, datab, sizeof(float) * count);
	if (err != CL_SUCCESS) {
		ocl_perror("copy datab", err);
		retval = -1;
		goto cleanup;
	}
	err = ocl_set_kernel_args(kernel, args);
	if (err != CL_SUCCESS) {
		ocl_perror("set kernel args", err);
		retval = -1;
		goto cleanup;
	}
	
	local = 1024;
	global = count;
	err = clEnqueueNDRangeKernel(environ.commands, kernel, 1, NULL,
				&global, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		ocl_perror("enqueue kernel", err);
		retval = -1;
		goto cleanup;
	}
	clFinish(environ.commands);

	err = clEnqueueReadBuffer(environ.commands, d_result, CL_TRUE, 0,
				sizeof(float) * count, result, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		ocl_perror("read result", err);
		retval = -1;
		goto cleanup;
	}
	
	correct = 0;
	for (i = 0; i < count; i++) {
		if (result[i] == dataa[i] * datab[i])
			correct++;
	}
	printf("%d correct, %d incorrect\n", correct, count - correct);

cleanup:
	if (d_result)
		clReleaseMemObject(d_result);
	if (d_dataa)
		clReleaseMemObject(d_dataa);
	if (d_datab)
		clReleaseMemObject(d_datab);
	if (kernel)
		clReleaseKernel(kernel);
	if (program)
		clReleaseProgram(program);
	ocl_destroy_environ(&environ);

	return retval;
}
