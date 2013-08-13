#ifndef __OCL_HELPERS_H__
#define __OCL_HELPERS_H__

#include <CL/opencl.h>

typedef struct ocl_environ {
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
} ocl_environ;

typedef struct ocl_kernel_arg {
	void *data;
	size_t size;
} ocl_kernel_arg;

cl_int ocl_default_environ(struct ocl_environ *environ);
void ocl_destroy_environ(struct ocl_environ *environ);
void ocl_print_build_error(cl_program program, cl_device_id device_id);
cl_int ocl_compile_file(struct ocl_environ *environ, const char *fname);
void ocl_perror(const char *prefix, cl_int err);
const char *ocl_strerror(cl_int err);
cl_int ocl_alloc_output(struct ocl_environ *environ, cl_mem *buffer, size_t size);
cl_int ocl_copy_input(struct ocl_environ *environ, cl_mem *buffer,
			void *data, size_t size);
cl_int ocl_set_kernel_args(cl_kernel kernel, const ocl_kernel_arg *arguments);
void ocl_check_error(ocl_environ *environ, cl_int err, const char *prefix);

#endif
