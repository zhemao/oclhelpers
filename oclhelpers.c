#include "oclhelpers.h"
#include <stdio.h>
#include <stdlib.h>

cl_int ocl_default_environ(ocl_environ *environ)
{
	cl_int err;

	cl_platform_id platform_id;
	cl_uint num_platforms;
	cl_uint num_devices;
	cl_context_properties properties[3];

	// zero out everything to begin with
	environ->device_id = 0;
	environ->context = NULL;
	environ->commands = NULL;
	
	err = clGetPlatformIDs(1, &platform_id, &num_platforms);

	if (err != CL_SUCCESS)
		return err;
	if (num_platforms != 1)
		return -1;
	
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT,
				1, &environ->device_id, &num_devices);

	if (err != CL_SUCCESS)
		return err;
	if (num_devices != 1)
		return -1;
	
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties) platform_id;
	properties[2] = 0;
	
	environ->context = clCreateContext(properties, num_devices,
				&environ->device_id, NULL, NULL, &err);

	if (err != CL_SUCCESS)
		return err;
	
	environ->commands = clCreateCommandQueue(
				environ->context,
				environ->device_id,
				0, &err);

	if (err != CL_SUCCESS)
		return err;
	
	return CL_SUCCESS;
}

void ocl_destroy_environ(ocl_environ *environ)
{
	if (environ->commands)
		clReleaseCommandQueue(environ->commands);
	if (environ->context)
		clReleaseContext(environ->context);
}

void ocl_print_build_error(cl_program program, cl_device_id device_id)
{
	size_t log_len;
	char logbuf[2048];

	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				sizeof(logbuf), logbuf, &log_len);
	printf("%s\n", logbuf);
}

static char *slurp(const char *fname)
{
	FILE *f = fopen(fname, "r");
	char *fdata;
	int len;

	if (fseek(f, 0, SEEK_END) != 0)
		return NULL;
	
	if ((len = ftell(f)) < 0)
		return NULL;
	
	if (fseek(f, 0, SEEK_SET) != 0)
		return NULL;
	
	fdata = malloc(len + 1);
	if (fdata == NULL)
		return NULL;
	
	if (fread(fdata, 1, len, f) != len) {
		free(fdata);
		return NULL;
	}

	fdata[len] = '\0';
	return fdata;
}

cl_int ocl_compile_file(const char *fname, 
			ocl_environ *environ, 
			cl_program *program)
{
	size_t len = 0;
	cl_int err;
	char *source = slurp(fname);

	if (source == NULL)
		return CL_INVALID_VALUE;

	*program = clCreateProgramWithSource(environ->context, 1, 
					(const char **) &source,
					(const size_t *) &len, &err);
	free(source);

	if (err != CL_SUCCESS)
		return err;
	
	err = clBuildProgram(*program, 1, &environ->device_id, 
				NULL, NULL, NULL);
	
	if (err != CL_SUCCESS) {
		ocl_print_build_error(*program, environ->device_id);
		return err;
	}

	return CL_SUCCESS;
}

cl_int ocl_alloc_output(ocl_environ *environ, cl_mem *buffer, size_t size)
{
	cl_int err;
	*buffer = clCreateBuffer(environ->context, CL_MEM_WRITE_ONLY,
				size, NULL, &err);
	return err;
}

cl_int ocl_copy_input(ocl_environ *environ, cl_mem *buffer,
			void *data, size_t size)
{
	cl_int err;
	*buffer = clCreateBuffer(environ->context, CL_MEM_READ_ONLY,
				size, NULL, &err);
	if (err != CL_SUCCESS)
		return err;
	err = clEnqueueWriteBuffer(environ->commands, *buffer, CL_TRUE, 0,
				size, data, 0, NULL, NULL);
	return err;
}

cl_int ocl_set_kernel_args(cl_kernel kernel, const ocl_kernel_arg *arguments)
{
	cl_int err;
	int i;

	for (i = 0; arguments[i].data != NULL; i++) {
		err = clSetKernelArg(kernel, i, 
				arguments[i].size,
				arguments[i].data);
		if (err != CL_SUCCESS)
			return err;
	}
	return CL_SUCCESS;
}

void ocl_perror(const char *prefix, cl_int err)
{
	printf("%s: %s\n", prefix, ocl_strerror(err));
}

/**
 * Error messages taken from 
 * https://www.khronos.org/message_boards/showthread.php/5912-error-to-string
 */
const char *ocl_strerror(cl_int err)
{
	switch (err) {
        case CL_SUCCESS:
		return "Success!";
        case CL_DEVICE_NOT_FOUND:
		return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:
		return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:
		return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:
		return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:
		return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:
		return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:
		return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:
		return "Program build failure";
        case CL_MAP_FAILURE:
		return "Map failure";
        case CL_INVALID_VALUE:
		return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:
		return "Invalid device type";
        case CL_INVALID_PLATFORM:
		return "Invalid platform";
        case CL_INVALID_DEVICE:
		return "Invalid device";
        case CL_INVALID_CONTEXT:
		return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:
		return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:
		return "Invalid command queue";
        case CL_INVALID_HOST_PTR:
		return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:
		return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:
		return "Invalid image size";
        case CL_INVALID_SAMPLER:
		return "Invalid sampler";
        case CL_INVALID_BINARY:
		return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:
		return "Invalid build options";
        case CL_INVALID_PROGRAM:
		return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:
		return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:
		return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:
		return "Invalid kernel definition";
        case CL_INVALID_KERNEL:
		return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:
		return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:
		return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:
		return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:
		return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:
		return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:
		return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:
		return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:
		return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:
		return "Invalid event wait list";
        case CL_INVALID_EVENT:
		return "Invalid event";
        case CL_INVALID_OPERATION:
		return "Invalid operation";
        case CL_INVALID_GL_OBJECT:
		return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:
		return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:
		return "Invalid mip-map level";
        default:
		return "Unknown";
    }
}
