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

/* Set up an OpenCL environment using the default device found by your
 * OpenCL implementation. Sets up device_id, context, and commands in environ. */
cl_int ocl_default_environ(struct ocl_environ *environ);

/* Destroy an environment initialized by ocl_default_environ.
 * This does not free the memory associated with the struct itself.
 * If you allocated dynamically, you must free it yourself */
void ocl_destroy_environ(struct ocl_environ *environ);

/* Print the error log from building an OpenCL program */
void ocl_print_build_error(cl_program program, cl_device_id device_id);

/* Load and build an opencl program from a file. 
 * The resulting program will be set in environ->program */
cl_int ocl_compile_file(struct ocl_environ *environ, const char *fname);

/* Returns an error message for the opencl error code */
const char *ocl_strerror(cl_int err);

/* Prints the error message for the error code prefixed with "$prefix:" */
void ocl_perror(const char *prefix, cl_int err);

/* Allocates a memory buffer for output */
cl_int ocl_alloc_output(struct ocl_environ *environ, cl_mem *buffer, size_t size);

/* Allocates a memory buffer for input and copies the data from the data
 * pointer to the newly allocated buffer.
 * The size is specified in bytes. */
cl_int ocl_copy_input(struct ocl_environ *environ, cl_mem *buffer,
			void *data, size_t size);

/* Sets a list of arguments into the kernel.
 * The list should be terminated by an ocl_kernel_arg where the data
 * pointer is NULL. */
cl_int ocl_set_kernel_args(cl_kernel kernel, const ocl_kernel_arg *arguments);

/* If err is not CL_SUCCESS, this function will print out an error
 * message using ocl_perror, destroy the environ, and then exit abnormally */
void ocl_check_error(ocl_environ *environ, cl_int err, const char *prefix);

#endif
