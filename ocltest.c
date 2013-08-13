#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "oclhelpers.h"

int verify_vector_mult(float *result, float *dataa, float *datab, 
			unsigned int count)
{
	int i;
	int correct = 0;

	for (i = 0; i < count; i++) {
		if (result[i] == dataa[i] * datab[i])
			correct++;
	}

	return correct;
}

int verify_matrix_mult(float *result, float *mata, float *matb, int matdim)
{
	int i, j, k;
	int correct = 0;
	float sum, vala, valb;
	
	for (i = 0; i < matdim; i++) {
		for (j = 0; j < matdim; j++) {
			sum = 0;
			for (k = 0; k < matdim; k++) {
				vala = mata[i * matdim + k];
				valb = matb[k * matdim + j];
				sum += vala * valb;
			}
			if (sum == result[i * matdim + j])
				correct++;
		}
	}

	return correct;
}

int main(void)
{
	struct ocl_environ environ;
	unsigned int count = 65536;
	unsigned int matdim = 256;
	size_t vec_local, vec_global;
	size_t mat_local[2], mat_global[2];
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

	ocl_kernel_arg vec_args[5] = {
		{&d_result, sizeof(cl_mem)},
		{&d_dataa, sizeof(cl_mem)},
		{&d_datab, sizeof(cl_mem)},
		{&count, sizeof(unsigned int)},
		{NULL, 0}};

	ocl_kernel_arg mat_args[7] = {
		{&d_result, sizeof(cl_mem)},
		{&d_dataa, sizeof(cl_mem)},
		{&d_datab, sizeof(cl_mem)},
		{&matdim, sizeof(matdim)},
		{&matdim, sizeof(matdim)},
		{&matdim, sizeof(matdim)},
		{NULL, 0}};


	srandom(time(0));

	for (i = 0; i < count; i++) {
		dataa[i] = rand();
		datab[i] = rand();
	}

	err = ocl_default_environ(&environ);
	ocl_check_error(&environ, err, "setup environ");
	
	ocl_compile_file(&environ, "ocltest_kernels.cl");
	ocl_check_error(&environ, err, "compile program");


	kernel = clCreateKernel(environ.program, "vector_mult", &err);
	ocl_check_error(&environ, err, "create vector kernel");

	err = ocl_alloc_output(&environ, &d_result, sizeof(float) * count);
	ocl_check_error(&environ, err, "alloc result");
	
	err = ocl_copy_input(&environ, &d_dataa, dataa, sizeof(float) * count);
	ocl_check_error(&environ, err, "copy dataa");
	
	err = ocl_copy_input(&environ, &d_datab, datab, sizeof(float) * count);
	ocl_check_error(&environ, err, "copy datab");
	
	err = ocl_set_kernel_args(kernel, vec_args);
	ocl_check_error(&environ, err, "set vector kernel args");
	
	vec_local = 1024;
	vec_global = count;
	err = clEnqueueNDRangeKernel(environ.commands, kernel, 1, NULL,
				&vec_global, &vec_local, 0, NULL, NULL);
	ocl_check_error(&environ, err, "enqueue vector kernel");
	clFinish(environ.commands);

	err = clEnqueueReadBuffer(environ.commands, d_result, CL_TRUE, 0,
				sizeof(float) * count, result, 0, NULL, NULL);
	ocl_check_error(&environ, err, "copy vector result");
	
	correct = verify_vector_mult(result, dataa, datab, count);
	printf("Vector: %d correct, %d incorrect\n", correct, count - correct);
	
	clReleaseKernel(kernel);

	kernel = clCreateKernel(environ.program, "matrix_mult", &err);
	ocl_check_error(&environ, err, "create matrix kernel");

	err = ocl_set_kernel_args(kernel, mat_args);
	ocl_check_error(&environ, err, "set matrix kernel args");

	mat_local[0] = 32;
	mat_local[1] = 32;
	mat_global[0] = matdim;
	mat_global[1] = matdim;
	
	err = clEnqueueNDRangeKernel(environ.commands, kernel, 2, NULL,
				mat_global, mat_local, 0, NULL, NULL);
	ocl_check_error(&environ, err, "enqueue matrix kernel");
	clFinish(environ.commands);

	err = clEnqueueReadBuffer(environ.commands, d_result, CL_TRUE, 0,
				sizeof(float) * count, result, 0, NULL, NULL);

	correct = verify_matrix_mult(result, dataa, datab, matdim);
	printf("Matrix: %d correct, %d incorrect\n", correct, count - correct);

	clReleaseMemObject(d_result);
	clReleaseMemObject(d_dataa);
	clReleaseMemObject(d_datab);
	clReleaseKernel(kernel);
	ocl_destroy_environ(&environ);

	return retval;
}
